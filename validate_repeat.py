import torch
import os

import argparse
from finetrainers.buffer_config.util import parse_partition_string

def parse_args():
    parser = argparse.ArgumentParser(description="Video generation validation script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="cogvideox",
        required=False,
        help="Name of model to use for validation"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/data/kinamkim/checkpoint/CogVideoX-5b",
        required=False,
        help="ID of model to use for validation"
    )
    parser.add_argument(
        "--latent_partition_mode",
        type=str,
        default="c1b3t9",
        required=False,
        help="Latent partition mode to use for validation"
    )
    parser.add_argument(
        "--lora_weight_path",
        type=str,
        required=False,
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset to use for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Width of the video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the video"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=30,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=30,
        help="Number of times to repeat each sample with increasing seed (default: 30)"
    )
    return parser.parse_args()


args = parse_args()

if __name__ == "__main__":
    with torch.no_grad():
        model_id = args.model_id
        if args.model_name == "cogvideox":
            from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
            from diffusers.utils import export_to_video, load_video
            from finetrainers.models.cogvideox.base_specification import process_video, custom_call

            pipe = CogVideoXPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            ).to("cuda")
            pipe.transformer.to("cpu")
            pipe.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer", torch_dtype=torch.bfloat16
            ).to("cuda")
            # pipe.enable_model_cpu_offload(device="cuda")
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            if args.lora_weight_path:
                pipe.load_lora_weights(args.lora_weight_path, adapter_name="cogvideox-lora")
                pipe.set_adapters(["cogvideox-lora"], [0.75])
            else:
                args.lora_weight_path = "outputs/base/dummy.safetensors"
        elif args.model_name == "wan":
            import torch
            from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
            from diffusers.utils import export_to_video
            from finetrainers.models.wan.base_specification import process_video, custom_call

            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            pipe.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
            pipe.to("cuda")

            if args.lora_weight_path:
                pipe.load_lora_weights(args.lora_weight_path, adapter_name="wan-lora")
                pipe.set_adapters(["wan-lora"], [0.75])
            else:
                args.lora_weight_path = "outputs/base/dummy.safetensors"

        # Create validation_videos directory in the same folder as lora_weight_path file
        lora_dir = os.path.dirname(args.lora_weight_path) if args.lora_weight_path else "outputs/base"
        savedir = os.path.join(lora_dir, "validation_videos")
        if args.latent_partition_mode:
            savedir = os.path.join(savedir, args.latent_partition_mode)
        else:
            savedir = os.path.join(savedir, "None")
        dataset_name = "/".join(args.dataset_dir.split("/")[-2:])
        savedir = os.path.join(savedir, dataset_name)
        os.makedirs(savedir, exist_ok=True)
        
        video_dir = os.path.join(args.dataset_dir, "videos")
        prompt_path = os.path.join(args.dataset_dir, "prompt.txt")
        with open(prompt_path, "r") as f:
            prompts = f.readlines()
        
        # Get and sort video files by their numeric names
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        video_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        # Duplicate first items due to the first video processing error in wan model
        if args.model_name == "wan":
            prompts = [prompts[0]] + prompts
            video_files = [video_files[0]] + video_files
        
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        neg_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        for i, prompt in enumerate(prompts[:args.num_videos+1]):
            pipe.to("cuda")
            print(f"Generating video {i+1}: {prompt[:100]}...")
            video_path = os.path.join(video_dir, video_files[i])
            print(f"video_path: {video_path}")
            from diffusers.utils import load_video
            video = load_video(video_path)

            # Determine repeat count and seed list
            if args.model_name == "wan" and i == 0:
                repeat_count = 1
            else:
                repeat_count = args.repeat

            for rep in range(repeat_count):
                seed = args.seed + rep * 10
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                init_latents = process_video(pipe, 
                                             video, 
                                             torch.bfloat16, 
                                             generator, 
                                             args.height, 
                                             args.width,
                                             args.latent_partition_mode)
                if args.latent_partition_mode == None:
                    init_latents = None
                # pipe.enable_model_cpu_offload()

                import types
                pipe.custom_call = types.MethodType(custom_call, pipe)
                video = pipe.custom_call(prompt, 
                             negative_prompt=neg_prompt,
                             generator=generator, 
                             width=args.width, 
                             height=args.height, 
                             num_frames=49,
                             latents=init_latents,
                             latent_partition_mode=args.latent_partition_mode).frames[0]
                if args.model_name == "wan" and i == 0:
                    output_name = f"dummy_output_seed_{seed}.mp4"
                else:
                    output_name = f"output_{i}_seed_{seed}.mp4"
                export_to_video(video, os.path.join(savedir, output_name))
                print(f"saved at {os.path.join(savedir, output_name)}")