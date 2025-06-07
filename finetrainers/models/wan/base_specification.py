import functools
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.processors import ProcessorMixin, T5Processor
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import get_non_null_items, safetensors_torch_save_function
from finetrainers.buffer_config.util import parse_partition_string

logger = get_logger()


class WanLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - latents_mean: The channel-wise mean of the latent space.
            - latents_std: The channel-wise standard deviation of the latent space.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 3

    def forward(
        self,
        vae: AutoencoderKLWan,
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
        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            # TODO(aryan): refactor in diffusers to have use_slicing attribute
            # if vae.use_slicing and video.shape[0] > 1:
            #     encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
            #     moments = torch.cat(encoded_slices)
            # else:
            #     moments = vae._encode(video)
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)

        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}


class WanImageConditioningLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - latents_mean: The channel-wise mean of the latent space.
            - latents_std: The channel-wise standard deviation of the latent space.
            - mask: The conditioning frame mask for the input image/video.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 4

    def forward(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        num_frames = video.size(2)
        if not self.use_last_frame:
            first_frame, remaining_frames = video[:, :, :1], video[:, :, 1:]
            video = torch.cat([first_frame, torch.zeros_like(remaining_frames)], dim=2)
        else:
            first_frame, remaining_frames, last_frame = video[:, :, :1], video[:, :, 1:-1], video[:, :, -1:]
            video = torch.cat([first_frame, torch.zeros_like(remaining_frames), last_frame], dim=2)

        # Image conditioning uses argmax sampling, so we use "mode" here
        if compute_posterior:
            latents = vae.encode(video).latent_dist.mode()
            latents = latents.to(dtype=dtype)
        else:
            # TODO(aryan): refactor in diffusers to have use_slicing attribute
            # if vae.use_slicing and video.shape[0] > 1:
            #     encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
            #     moments = torch.cat(encoded_slices)
            # else:
            #     moments = vae._encode(video)
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)

        temporal_downsample = 2 ** sum(vae.temperal_downsample) if getattr(self, "vae", None) else 4
        mask = latents.new_ones(latents.shape[0], 1, num_frames, latents.shape[3], latents.shape[4])
        if not self.use_last_frame:
            mask[:, :, 1:] = 0
        else:
            mask[:, :, 1:-1] = 0
        first_frame_mask = mask[:, :, :1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=temporal_downsample)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
        mask = mask.view(latents.shape[0], -1, temporal_downsample, latents.shape[3], latents.shape[4])
        mask = mask.transpose(1, 2)

        return {
            self.output_names[0]: latents,
            self.output_names[1]: latents_mean,
            self.output_names[2]: latents_std,
            self.output_names[3]: mask,
        }


class WanImageEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encoding image conditioning for Wan I2V training.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - image_embeds: The CLIP vision model image embeddings of the input image.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 1

    def forward(
        self,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = image_encoder.device
        dtype = image_encoder.dtype
        last_image = None

        # We know the image here is in the range [-1, 1] (probably a little overshot if using bilinear interpolation), but
        # the processor expects it to be in the range [0, 1].
        image = image if video is None else video[:, 0]  # [B, F, C, H, W] -> [B, C, H, W] (take first frame)
        image = FF.normalize(image, min=0.0, max=1.0, dim=1)
        assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor"

        if self.use_last_frame:
            last_image = image if video is None else video[:, -1]
            last_image = FF.normalize(last_image, min=0.0, max=1.0, dim=1)
            image = torch.stack([image, last_image], dim=0)

        image = image_processor(images=image.float(), do_rescale=False, do_convert_rgb=False, return_tensors="pt")
        image = image.to(device=device, dtype=dtype)
        image_embeds = image_encoder(**image, output_hidden_states=True)
        image_embeds = image_embeds.hidden_states[-2]
        return {self.output_names[0]: image_embeds}


class WanModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
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

        use_last_frame = self.transformer_config.get("pos_embed_seq_len", None) is not None

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "__drop__"])]
        if latent_model_processors is None:
            latent_model_processors = [WanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"])]

        if self.transformer_config.get("image_dim", None) is not None:
            latent_model_processors.append(
                WanImageConditioningLatentEncodeProcessor(
                    ["latent_condition", "__drop__", "__drop__", "latent_condition_mask"],
                    use_last_frame=use_last_frame,
                )
            )
            latent_model_processors.append(
                WanImageEncodeProcessor(["encoder_hidden_states_image"], use_last_frame=use_last_frame)
            )

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLWan.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        models = {"vae": vae}
        if self.transformer_config.get("image_dim", None) is not None:
            # TODO(aryan): refactor the trainer to be able to support these extra models from CLI args more easily
            image_encoder = CLIPVisionModel.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=torch.bfloat16
            )
            image_processor = CLIPImageProcessor.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_processor"
            )
            models["image_encoder"] = image_encoder
            models["image_processor"] = image_processor

        return models

    def load_diffusion_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = WanTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = WanTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[UMT5EncoderModel] = None,
        transformer: Optional[WanTransformer3DModel] = None,
        vae: Optional[AutoencoderKLWan] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        image_encoder: Optional[CLIPVisionModel] = None,
        image_processor: Optional[CLIPImageProcessor] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Union[WanPipeline, WanImageToVideoPipeline]:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }
        components = get_non_null_items(components)

        if self.transformer_config.get("image_dim", None) is not None:
            pipe = WanPipeline.from_pretrained(
                self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
            )
        else:
            pipe = WanPipeline.from_pretrained(
                self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
            )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        if not training:
            pipe.transformer.to(self.transformer_dtype)

        # TODO(aryan): add support in diffusers
        # if enable_slicing:
        #     pipe.vae.enable_slicing()
        # if enable_tiling:
        #     pipe.vae.enable_tiling()
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        caption: str,
        max_sequence_length: int = 512,
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
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLWan,
        image_encoder: Optional[CLIPVisionModel] = None,
        image_processor: Optional[CLIPImageProcessor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
            "image": image,
            "video": video,
            "generator": generator,
            # We must force this to False because the latent normalization should be done before
            # the posterior is computed. The VAE does not handle this any more:
            # https://github.com/huggingface/diffusers/pull/10998
            "compute_posterior": False,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    def forward(
        self,
        transformer: WanTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        latent_partition_mode: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        compute_posterior = False  # See explanation in prepare_latents
        latent_condition = latent_condition_mask = None

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)
        else:
            latents = latent_model_conditions.pop("latents")
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=generator)

            if latent_condition is not None:
                mu, logvar = torch.chunk(latent_condition, 2, dim=1)
                mu = self._normalize_latents(mu, latents_mean, latents_std)
                logvar = self._normalize_latents(logvar, latents_mean, latents_std)
                latent_condition = torch.cat([mu, logvar], dim=1)

                posterior = DiagonalGaussianDistribution(latent_condition)
                latent_condition = posterior.mode()

            del posterior

        noise = torch.zeros_like(latents).normal_(generator=generator)
        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)

        latent_partition_config = parse_partition_string(latent_partition_mode)
        condition_count = latent_partition_config["condition"]
        buffer_count = latent_partition_config["buffer"]
        target_count = latent_partition_config["target"]

        noisy_latents[:, :, :condition_count] = latents[:, :, :condition_count]
        if buffer_count == 3:
            mask_075 = (sigmas > 0.75).view(-1, 1, 1, 1, 1)
            mask_050 = (sigmas > 0.5).view(-1, 1, 1, 1, 1)
            mask_025 = (sigmas > 0.25).view(-1, 1, 1, 1, 1)
            
            noisy_latents_075 = FF.flow_match_xt(latents[:, :, condition_count+2:condition_count+3], noise[:, :, condition_count+2:condition_count+3], 0.75)
            noisy_latents_050 = FF.flow_match_xt(latents[:, :, condition_count+1:condition_count+2], noise[:, :, condition_count+1:condition_count+2], 0.5)
            noisy_latents_025 = FF.flow_match_xt(latents[:, :, condition_count:condition_count+1], noise[:, :, condition_count:condition_count+1], 0.25)
            
            noisy_latents[:, :, condition_count+2:condition_count+3] = torch.where(mask_075, noisy_latents_075, noisy_latents[:, :, condition_count+2:condition_count+3])
            noisy_latents[:, :, condition_count+1:condition_count+2] = torch.where(mask_050, noisy_latents_050, noisy_latents[:, :, condition_count+1:condition_count+2])
            noisy_latents[:, :, condition_count:condition_count+1] = torch.where(mask_025, noisy_latents_025, noisy_latents[:, :, condition_count:condition_count+1])
        else:
            raise ValueError(f"Buffer count {buffer_count} not supported")
        timesteps = (sigmas.flatten() * 1000.0).long()

        if self.transformer_config.get("image_dim", None) is not None:
            noisy_latents = torch.cat([noisy_latents, latent_condition_mask, latent_condition], dim=1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            return_dict=False,
        )[0]
        target = FF.flow_match_target(noise, latents)

        return pred, target, sigmas

    def validation(
        self,
        pipeline: Union[WanPipeline, WanImageToVideoPipeline],
        prompt: str,
        image: Optional[PIL.Image.Image] = None,
        last_image: Optional[PIL.Image.Image] = None,
        video: Optional[List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        latent_partition_mode: Optional[str] = None,
        **kwargs,
    ) -> List[ArtifactType]:

        init_latents = process_video(pipeline, video, pipeline.dtype, generator, height, width, latent_partition_mode)
        pipeline.custom_call = types.MethodType(custom_call, pipeline)

        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
            "latents": init_latents,
            "latent_partition_mode": latent_partition_mode,
        }
        if self.transformer_config.get("image_dim", None) is not None:
            if image is None and video is None:
                raise ValueError("Either image or video must be provided for Wan I2V validation.")
            image = image if image is not None else video[0]
            generation_kwargs["image"] = image
        if self.transformer_config.get("pos_embed_seq_len", None) is not None:
            last_image = last_image if last_image is not None else image if video is None else video[-1]
            generation_kwargs["last_image"] = last_image
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline.custom_call(**generation_kwargs).frames[0]
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
        pipeline_cls = (
            WanImageToVideoPipeline if self.transformer_config.get("image_dim", None) is not None else WanPipeline
        )
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            pipeline_cls.save_lora_weights(
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
        transformer: WanTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = WanTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

import types
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
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
    from diffusers.utils import load_video
    from diffusers.pipelines.wan.pipeline_wan_video2video import retrieve_latents
    from diffusers.utils.torch_utils import randn_tensor
    video = pipe.video_processor.preprocess_video(video, height=height, width=width)
    video = video.to("cuda", dtype=pipe.vae.dtype)
    
    video_latents = retrieve_latents(pipe.vae.encode(video))
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(pipe.device, dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        pipe.device, dtype
    )

    init_latents = (video_latents - latents_mean) * latents_std

    init_latents = init_latents.to(pipe.device)

    noise = randn_tensor(init_latents.shape, generator=generator, device=pipe.device, dtype=dtype)
    latent_partition_config = parse_partition_string(latent_partition_mode)
    condition_count = latent_partition_config["condition"]
    buffer_count = latent_partition_config["buffer"]
    target_count = latent_partition_config["target"]

    timesteps = pipe.scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
    scheduler = pipe.scheduler
    n_timesteps = timesteps.shape[0]
    t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
    t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
    t_75 = timesteps[int(n_timesteps * (1 - 0.75))]
    
    init_latents[:, :, condition_count] = scheduler.add_noise(init_latents[:, :, condition_count], noise[:, :, condition_count], torch.tensor([t_25]))
    init_latents[:, :, condition_count+1] = scheduler.add_noise(init_latents[:, :, condition_count+1], noise[:, :, condition_count+1], torch.tensor([t_50]))
    init_latents[:, :, condition_count+2] = scheduler.add_noise(init_latents[:, :, condition_count+2], noise[:, :, condition_count+2], torch.tensor([t_75]))
    init_latents[:, :, condition_count+3:] = noise[:, :, condition_count+3:]
    init_latents = init_latents.to(pipe.device)
    return init_latents

@torch.no_grad()
def retrieve_video(pipe, init_latents):
    latents = init_latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    return video

@torch.no_grad()
def custom_call(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    latent_partition_mode: Optional[str] = None,
):

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    if num_frames % self.vae_scale_factor_temporal != 1:
        logger.warning(
            f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = self._execution_device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    timesteps = self.scheduler.timesteps 
    scheduler = self.scheduler
    n_timesteps = timesteps.shape[0]
    #t_100 = timesteps[0]
    t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
    t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
    t_75 = timesteps[int(n_timesteps * (1 - 0.75))]
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            latent_partition_config = parse_partition_string(latent_partition_mode)
            condition_count = latent_partition_config["condition"]
            buffer_count = latent_partition_config["buffer"]
            target_count = latent_partition_config["target"]
            noise_pred[:, :, :condition_count] = 0
            if buffer_count == 3:
                if t > t_25:
                    noise_pred[:, :, condition_count] = 0
                if t > t_50:
                    noise_pred[:, :, condition_count+1] = 0
                if t > t_75:
                    noise_pred[:, :, condition_count+2] = 0
            else:
                raise ValueError(f"Buffer count {buffer_count} not supported")

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None

    if not output_type == "latent":
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return WanPipelineOutput(frames=video)