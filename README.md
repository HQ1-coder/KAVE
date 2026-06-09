# KAVE-K-Way-Angular-Video-Steganography-with-Temporal-Resilience-for-Deterministic-Video-Diffusion

# KAVE

Official implementation of **KAVE**, K-Way Angular Video Steganography with Temporal Resilience for Deterministic Video Diffusion. KAVE is built on [Wan2.1](https://github.com/Wan-Video/Wan2.1).

---

## 1. Installation

```bash
git clone <this-repo-url>
cd KAVE
pip install -r requirements.txt
```

---

## 2. Model Download

Download the Wan2.1-T2V-1.3B model weights into `./Wan2.1-T2V-1.3B`:

## 3. Generate Stego Video

```bash
python sender.py \
  --task t2v-1.3B --size 832*480 --frame_num 81 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --offload_model True \
  --prompt "It's drizzling in the sky, and on the country lane, an orange cat is riding a bicycle. To avoid the rain, it puts a lotus leaf on its head." \
  --base_seed 99 \
  --sample_solver unipc --sample_steps 50 --sample_shift 5.0 --sample_guide_scale 5.0 \
  --mask1 0.32 --mask2 0.925 --add_cfg 16.0 --sector_margin_deg 40.0 \
  --reference_mode base_cosine \
  --symbol_redundancy 10 \
  --mask1_mode clean --mask2_mode clean_tiebreak \
  --embedding_strength 5.0 \
  --channel_coding repetition --redundancy_schedule uniform \
  --pair_selection_mode full \
  --save_file ./stego_output.mp4
```

---

## 4. Extract Message 

```bash
python receiver.py \
  --task t2v-1.3B --size 832*480 --frame_num 81 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --offload_model True \
  --prompt "It's drizzling in the sky, and on the country lane, an orange cat is riding a bicycle. To avoid the rain, it puts a lotus leaf on its head." \
  --base_seed 99 \
  --sample_solver unipc --sample_steps 50 --sample_shift 5.0 --sample_guide_scale 5.0 \
  --mask1 0.32 --mask2 0.925 --add_cfg 16.0 --sector_margin_deg 40.0 \
  --reference_mode base_cosine \
  --symbol_redundancy 10 \
  --mask1_mode clean --mask2_mode clean_tiebreak \
  --embedding_strength 5.0 \
  --channel_coding repetition --redundancy_schedule uniform \
  --pair_selection_mode full \
  --base_replay_mode clean \
  --video_path ./stego_output.mp4 \
  --message_save_file ./recovered_message.pt
```

---

## 5. Evaluate PSNR

Modify the video paths in `PSNR.py` and run:

```bash
python PSNR.py
```

---

## Acknowledgments

This code is developed based on the work:

> **LD-RoViS: Training-free Robust Video Steganography for Deterministic Latent Diffusion Model**
> Xiangkun Wang, Kejiang Chen, Lincong Li, Weiming Zhang, Nenghai Yu
> *NeurIPS 2025*

We thank the LD-RoViS authors for their open-source contribution.

We also thank [Wan2.1](https://github.com/Wan-Video/Wan2.1) for the video generation backbone.

## License

Apache 2.0 License. See [LICENSE.txt](LICENSE.txt).
