import functools
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from PIL.Image import Image
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.models.utils import DiagonalGaussianDistribution
from finetrainers.processors import ProcessorMixin, T5Processor
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import _enable_vae_memory_optimizations, get_non_null_items, safetensors_torch_save_function
from finetrainers.buffer_config.util import parse_partition_string

from .utils import prepare_rotary_positional_embeddings


logger = get_logger()


class CogVideoXLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the CogVideoX VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKLCogVideoX,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            if vae.use_slicing and video.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
                moments = torch.cat(encoded_slices)
            else:
                moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]
        return {self.output_names[0]: latents}


class CogVideoXModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "THUDM/CogVideoX-5b",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "prompt_attention_mask"])]
        if latent_model_processors is None:
            latent_model_processors = [CogVideoXLatentEncodeProcessor(["latents"])]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (1, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLCogVideoX.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        scheduler = CogVideoXDDIMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler", **common_kwargs
        )

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[T5Tokenizer] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        transformer: Optional[CogVideoXTransformer3DModel] = None,
        vae: Optional[AutoencoderKLCogVideoX] = None,
        scheduler: Optional[CogVideoXDDIMScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> CogVideoXPipeline:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        pipe = CogVideoXPipeline.from_pretrained(
            self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
        if not training:
            pipe.transformer.to(self.transformer_dtype)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        caption: str,
        max_sequence_length: int = 226,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        conditions.pop("prompt_attention_mask", None)
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLCogVideoX,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image": image,
            "video": video,
            "generator": generator,
            "compute_posterior": compute_posterior,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    def forward(
        self,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDDIMScheduler,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        latent_partition_mode = "c1b3t9",
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # Just hardcode for now. In Diffusers, we will refactor such that RoPE would be handled within the model itself.
        VAE_SPATIAL_SCALE_FACTOR = 8
        rope_base_height = self.transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
        rope_base_width = self.transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
        patch_size = self.transformer_config.patch_size
        patch_size_t = getattr(self.transformer_config, "patch_size_t", None)

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
        else:
            posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("latents"), _dim=2)
            latents = posterior.sample(generator=generator)
            del posterior

        if not getattr(self.vae_config, "invert_scale_latents", False):
            latents = latents * self.vae_config.scaling_factor

        if patch_size_t is not None:
            latents = self._pad_frames(latents, patch_size_t)

        timesteps = (sigmas.flatten() * 1000.0).long()

        noise = torch.zeros_like(latents).normal_(generator=generator) # [B, F, C, H, W]
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        latent_partition_config = parse_partition_string(latent_partition_mode)
        assert latent_partition_config["condition"] + latent_partition_config["buffer"] + latent_partition_config["target"] == latents.shape[1], f"Latent partition config {latent_partition_mode} should sum to {latents.shape[1]}"

        condition_count = latent_partition_config["condition"]
        buffer_count = latent_partition_config["buffer"]
        target_count = latent_partition_config["target"]

        noisy_latents[:, :condition_count] = latents[:, :condition_count]
        if buffer_count == 3:
            whole_timesteps = scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
            n_timesteps = whole_timesteps.shape[0]
            #t_100 = timesteps[0]
            t_25 = whole_timesteps[int(n_timesteps * (1 - 0.25))]
            t_50 = whole_timesteps[int(n_timesteps * (1 - 0.5))]
            t_75 = whole_timesteps[int(n_timesteps * (1 - 0.75))]

            mask_075 = (timesteps > t_75).view(-1, 1, 1, 1, 1)
            mask_050 = (timesteps > t_50).view(-1, 1, 1, 1, 1)
            mask_025 = (timesteps > t_25).view(-1, 1, 1, 1, 1)
            
            noisy_latents_075 = scheduler.add_noise(latents[:, condition_count+2:condition_count+3], noise[:, condition_count+2:condition_count+3], torch.tensor([t_75]))
            noisy_latents_050 = scheduler.add_noise(latents[:, condition_count+1:condition_count+2], noise[:, condition_count+1:condition_count+2], torch.tensor([t_50]))
            noisy_latents_025 = scheduler.add_noise(latents[:, condition_count:condition_count+1], noise[:, condition_count:condition_count+1], torch.tensor([t_25]))
            
            noisy_latents[:, condition_count+2:condition_count+3] = torch.where(mask_075, noisy_latents_075, noisy_latents[:, condition_count+2:condition_count+3])
            noisy_latents[:, condition_count+1:condition_count+2] = torch.where(mask_050, noisy_latents_050, noisy_latents[:, condition_count+1:condition_count+2])
            noisy_latents[:, condition_count:condition_count+1] = torch.where(mask_025, noisy_latents_025, noisy_latents[:, condition_count:condition_count+1])
        else:
            raise ValueError(f"Buffer count is only supported for 3 frames")

        batch_size, num_frames, num_channels, height, width = latents.shape
        ofs_emb = (
            None
            if getattr(self.transformer_config, "ofs_embed_dim", None) is None
            else latents.new_full((batch_size,), fill_value=2.0)
        )

        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * VAE_SPATIAL_SCALE_FACTOR,
                width=width * VAE_SPATIAL_SCALE_FACTOR,
                num_frames=num_frames,
                vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                attention_head_dim=self.transformer_config.attention_head_dim,
                device=transformer.device,
                base_height=rope_base_height,
                base_width=rope_base_width,
            )
            if self.transformer_config.use_rotary_positional_embeddings
            else None
        )

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
        latent_model_conditions["image_rotary_emb"] = image_rotary_emb
        latent_model_conditions["ofs"] = ofs_emb

        velocity = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            return_dict=False,
        )[0]
        # For CogVideoX, the transformer predicts the velocity. The denoised output is calculated by applying the same
        # code paths as scheduler.get_velocity(), which can be confusing to understand.
        pred = scheduler.get_velocity(velocity, noisy_latents, timesteps)
        target = latents

        return pred, target, sigmas

    def validation(
        self,
        pipeline: CogVideoXPipeline,
        prompt: str,
        image: Optional[Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        # TODO(aryan): add support for more parameters
        if image is not None:
            pipeline = CogVideoXImageToVideoPipeline.from_pipe(pipeline)

        generation_kwargs = {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline(**generation_kwargs).frames[0]
        return [VideoArtifact(value=video)]

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            CogVideoXPipeline.save_lora_weights(
                directory,
                transformer_state_dict,
                save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                safe_serialization=True,
            )
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
        self,
        directory: str,
        transformer: CogVideoXTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = CogVideoXTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _pad_frames(latents: torch.Tensor, patch_size_t: int) -> torch.Tensor:
        num_frames = latents.size(1)
        additional_frames = patch_size_t - (num_frames % patch_size_t)
        if additional_frames > 0:
            last_frame = latents[:, -1:]
            padding_frames = last_frame.expand(-1, additional_frames, -1, -1, -1)
            latents = torch.cat([latents, padding_frames], dim=1)
        return latents

import math
import types
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

@torch.no_grad()
def process_video(pipe, video, dtype, generator, height, width, latent_partition_mode):
    if pipe.device != "cuda":
        generator = None
    if latent_partition_mode == None:
        return None
    from diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video import retrieve_latents
    from diffusers.utils.torch_utils import randn_tensor
    video = pipe.video_processor.preprocess_video(video, height=height, width=width)
    video = video.to("cuda", dtype=dtype)
    
    video_latents = retrieve_latents(pipe.vae.encode(video))
    init_latents = video_latents.to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    init_latents = pipe.vae_scaling_factor_image * init_latents
    init_latents = init_latents.to(pipe.device)

    noise = randn_tensor(init_latents.shape, generator=generator, device=pipe.device, dtype=dtype)
    latent_partition_config = parse_partition_string(latent_partition_mode)
    condition_count = latent_partition_config["condition"]
    buffer_count = latent_partition_config["buffer"]
    target_count = latent_partition_config["target"]
    
    if buffer_count == 3:
        timesteps = pipe.scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
        scheduler = pipe.scheduler
        n_timesteps = timesteps.shape[0]

        t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
        t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
        t_75 = timesteps[int(n_timesteps * (1 - 0.75))]

        init_latents[:, condition_count] = scheduler.add_noise(init_latents[:, condition_count], noise[:, condition_count], torch.tensor([t_25]))
        init_latents[:, condition_count+1] = scheduler.add_noise(init_latents[:, condition_count+1], noise[:, condition_count+1], torch.tensor([t_50]))
        init_latents[:, condition_count+2] = scheduler.add_noise(init_latents[:, condition_count+2], noise[:, condition_count+2], torch.tensor([t_75]))
        init_latents[:, condition_count+3:] = noise[:, condition_count+3:]
    else:
        raise ValueError(f"Buffer count is only supported for 3 frames")
    
    init_latents = init_latents.to(pipe.device)
    return init_latents

@torch.no_grad()
def retrieve_video(pipe, init_latents,):
    init_latents = init_latents.to("cuda")
    video = pipe.decode_latents(init_latents)
    video = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]
    return video

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def custom_call(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
    latent_partition_mode: Optional[str] = None,
) -> Union[CogVideoXPipelineOutput, Tuple]:
    print("<Custom Call>")
    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds,
        negative_prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = self.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * self.vae_scale_factor_temporal

    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    timesteps = self.scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
    scheduler = self.scheduler
    n_timesteps = timesteps.shape[0]
    #t_100 = timesteps[0]
    t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
    t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
    t_75 = timesteps[int(n_timesteps * (1 - 0.75))]

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)                
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latent_partition_config = parse_partition_string(latent_partition_mode)
            condition_count = latent_partition_config["condition"]
            buffer_count = latent_partition_config["buffer"]
            target_count = latent_partition_config["target"]

            noise_pred[:, :condition_count] = 0
            if buffer_count == 3:
                if t > t_25:
                    noise_pred[:, condition_count:condition_count+1] = 0
                if t > t_50:
                    noise_pred[:, condition_count+1:condition_count+2] = 0
                if t > t_75:
                    noise_pred[:, condition_count+2:condition_count+3] = 0
            else:
                raise ValueError(f"Buffer count is only supported for 3 frames")

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None

    if not output_type == "latent":
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)