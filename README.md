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

## 🗺️Start Guide
## 📂 Dataset

- Download dataset zip file here: [Drive](tmp link)
- Unzip the dataset under `dataset/`, and make sure the structure matches `dataset/example/`.

- For custom datasets, follow the structure of `dataset/example/`.

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


