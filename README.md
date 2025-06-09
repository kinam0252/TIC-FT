# 🚀Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models✨

## 📑Paper
- Arxiv: [Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models](https://arxiv.org/abs/2506.00996)

## 🌐Project Page
- [TIC-FT Project Page](https://kinam0252.github.io/TIC-FT/)

# 🚀 Try It Yourself!

Follow these steps to easily test the I2V pipeline:

1. **Prepare Your Image**  
   Convert your face image into either **Cartoon** or **3D Animation** style with a white background using an image generation tool such as ChatGPT.

   <img src="https://github.com/user-attachments/assets/2282e710-f6fa-4bab-af38-547b476dc26b" width="480">

3. **Save the Image**  
   Save your generated image to:  
   `dataset/custom/{mode}/images`  
   - `{mode}` could be either `Cartoon` or `3DAnimation`.
   - By default, an example `1.png` is provided. You can:
     - Add new images as `2.png`, `3.png`, etc.
     - Or replace `1.png` directly.

4. **Convert Image to Reference Video**  
   Use the following script to duplicate the image into 49 frames and generate a condition video:
   ```bash
   python dataset/utils/make_video_by_copying_image.py {image_path}
   ```
   Save the generated condition video into: dataset/custom/{mode}/videos
5. **Prepare Dataset Files**
   - In dataset/custom/{mode}/videos.txt, list the relative video paths (one per line).
   - In dataset/custom/{mode}/prompt.txt, write the corresponding text prompts (one per line).

6. **Download Pretrained Weights**  
   Download the safetensors weights for your selected mode from:  
   [Google Drive](https://drive.google.com/drive/folders/1TXME89uReXw4VFFW5BmYrKHdfyfpAQAv?usp=drive_link)

7. **Run Inference**  
   Example command:
   ```bash
   python validate_repeat.py \
   --model_name wan \
   --model_id Wan2.1-T2V-14B-Diffusers \
   --lora_weight_path /data/kinamkim/TIC-FT/outputs/wan/3DAnimation/pytorch_lora_weights.safetensors \
   --latent_partition_mode c1b3t9 \
   --dataset_dir /data/kinamkim/dummy/TIC-FT/dataset/custom/3DAnimation

8. Now you have your own video featuring your character!
   

https://github.com/user-attachments/assets/ca34819f-52be-4b05-9cf0-747c902bb36a



https://github.com/user-attachments/assets/1fdee654-49e0-4481-b3cd-266fd7105f7b



## 🚧 Progress

### ✅ Completed
- [x] Implement I2V code on both CogVideoX and Wan

### 🔄 In Progress
- [ ] Prepare model weights for various I2V applications
- [ ] Implement V2V code for CogVideoX

### 🔜 Upcoming
- [ ] Implement remaining features: Multiple Conditions, Action Transfer, and Video Interpolation

## 🗺️Start Guide
### 🔗 Weights
- Download pretrained weights from here: [Drive](https://drive.google.com/drive/folders/1asL4g2mutM4AtR6ygXEgabfPszSRQ2iW?usp=drive_link)

## 🚀 Train

- For **CogVideoX**:  
  Example:
  ```bash
  scripts/cogvideox/I2V/train.sh

- For **Wan**:
  Example:
  ```bash
  scripts/wan/I2V/train.sh

## 🔎 Inference
```bash
python validate.py \
  --model_name wan \
  --model_id {checkpoint path} \
  --lora_weight_path {safetensors path} \
  --latent_partition_mode c1b3t9 \
  --dataset_dir {dataset dir}
```

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


