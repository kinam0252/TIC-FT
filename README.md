# 🚀Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models✨

## 📑Paper
- Arxiv: [Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models](https://arxiv.org/abs/2506.00996)

## 🌐Project Page
- [TIC-FT Project Page](https://kinam0252.github.io/TIC-FT/)

## 🚧 Progress

### ✅ Completed
- [x] Implement I2V code on both CogVideoX and Wan

### 🔄 In Progress
- [ ] Prepare model weights for various I2V applications
- [ ] Implement V2V code for CogVideoX

### 🔜 Upcoming
- [ ] Implement remaining features: Multiple Conditions, Action Transfer, and Video Interpolation

## 🎥 Video Examples

Below are example videos showcasing various applications of TIC-FT.

---

### 🖼️ I2V
https://github.com/user-attachments/assets/addbc3cb-1b9a-429b-a303-1b1d90f413d0

https://github.com/user-attachments/assets/35031954-bdbb-4fae-a816-7ac4d211fb5e

https://github.com/user-attachments/assets/cb646533-da91-475d-bbc3-c18927b0fb12

https://github.com/user-attachments/assets/17197e22-935a-4b15-86c5-913d9e0ff5bb

---

### 🔁 V2V
https://github.com/user-attachments/assets/a65ac52f-d0b9-4122-8de6-686225e7b618

https://github.com/user-attachments/assets/95329b8f-fcff-4a64-ac52-13091249c269

---

### 🖼️ Multiple Conditions
https://github.com/user-attachments/assets/621e6bdc-8d0c-457a-9d33-7f81a707946f

https://github.com/user-attachments/assets/29cb5bfe-a6d2-41b0-a9c8-3c1a5e5301a8

---

### 🎯 Action Transfer

https://github.com/user-attachments/assets/9cba035f-45ae-4f78-8dfc-b933e94a22af

---

### 🕰️ Keyframe Interpolation


https://github.com/user-attachments/assets/feed64a0-eb5d-416f-9721-489504d1cd0b




## 🗺️Start Guide
🧪**Diffusers-based codes**
   To run the test script, refer to the `inference.py` file in each folder. Below is an example using Mochi:
   
   ```python
   # inference.py
   import torch
   from diffusers import MochiPipeline
   from pipeline_stg_mochi import MochiSTGPipeline
   from diffusers.utils import export_to_video
   import os
   
   # Ensure the samples directory exists
   os.makedirs("samples", exist_ok=True)
   
   ckpt_path = "genmo/mochi-1-preview"
   # Load the pipeline
   pipe = MochiSTGPipeline.from_pretrained(ckpt_path, variant="bf16", torch_dtype=torch.bfloat16)
   
   # Enable memory savings
   # pipe.enable_model_cpu_offload()
   # pipe.enable_vae_tiling()
   pipe = pipe.to("cuda")
   
   #--------Option--------#
   prompt = "A close-up of a beautiful woman's face with colored powder exploding around her, creating an abstract splash of vibrant hues, realistic style."
   stg_applied_layers_idx = [34]
   stg_mode = "STG"
   stg_scale = 1.0 # 0.0 for CFG (default)
   do_rescaling = False # False (default)
   #----------------------#
   
   # Generate video frames
   frames = pipe(
       prompt, 
       height=480,
       width=480,
       num_frames=81,
       stg_applied_layers_idx=stg_applied_layers_idx,
       stg_scale=stg_scale,
       generator = torch.Generator().manual_seed(42),
       do_rescaling=do_rescaling,
   ).frames[0]
   
   # Construct the video filename
   if stg_scale == 0:
       video_name = f"CFG_rescale_{do_rescaling}.mp4"
   else:
       layers_str = "_".join(map(str, stg_applied_layers_idx))
       video_name = f"{stg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"
   
   # Save video to samples directory
   video_path = os.path.join("samples", video_name)
   export_to_video(frames, video_path, fps=30)
   
   print(f"Video saved to {video_path}")
   ```
   For details on memory efficiency, inference acceleration, and more, refer to the original pages below:
   - [Mochi](https://huggingface.co/genmo/mochi-1-preview)
   - [CogVideoX](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox)
   - [HunyuanVideo](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video)
   - [StableVideoDiffusion](https://huggingface.co/docs/diffusers/en/using-diffusers/svd)


## 🙏Acknowledgements
This project is built upon the following works:
- [finetrainers](https://github.com/a-r-r-o-w/finetrainers)

## 📖 BibTeX

```bibtex
@article{kim2025temporal,
  title={Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models},
  author={Kim, Kinam and Hyung, Junha and Choo, Jaegul},
  journal={arXiv preprint arXiv:2506.00996},
  year={2025}
}


