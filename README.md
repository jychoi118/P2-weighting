# P2 weighting (CVPR 2022)

This is the codebase for [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227).

This repository is heavily based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

P2 modifies the weighting scheme of the training objective function to improve sample quality. It encourages the diffusion model to focus on recovering signals from highly corrupted data, where the model learns global and perceptually rich concepts. Below figure shows the weighting schemes in terms of SNR.

![snr_weight](https://user-images.githubusercontent.com/36615789/161203299-8b02d76b-9c51-4529-8329-3ac08e9f3bc8.png)

## Pre-trained models

All models are trained at 256x256 resolution.

Here are the download links for each model checkpoint:

 * FFHQ baseline: [ffhq_baseline.pt](https://drive.google.com/file/d/17SR1lih6BxBxJhr8s1iOjhw540H3Pc_m/view?usp=sharing)
 * FFHQ ours: [ffhq_p2.pt](https://drive.google.com/file/d/1nlCPBqOqSeqAQ8F4noThbnmr4okriwTc/view?usp=sharing)
 * CelebA-HQ ours: [celebahq_p2.pt](https://drive.google.com/file/d/1ag8JnvKGKo6L6avO_dTFEBNWJo2jZIDF/view?usp=sharing)
 * CUB baseline: [cub_baseline.pt](https://drive.google.com/file/d/1Wv-hHL7bhGsWyCp3i-RK6YcodcRrD-dj/view?usp=sharing)
 * CUB ours: [cub_p2.pt](https://drive.google.com/file/d/13RKF9MLNR3zpMMNchW7JtsjktVMgS0rK/view?usp=sharing)
 * AFHQ-dog baseline: [afhq_baseline.pt](https://drive.google.com/file/d/1bv-xnJC1-qBg9ZlVdugsXLuHVsKj1RRE/view?usp=sharing)
 * AFHQ-dog ours: [afhq_p2.pt](https://drive.google.com/file/d/1f6_swzSPNJXs9FWf6AW585bv7dORzRWf/view?usp=sharing)
 * Flowers baseline: [flower_baseline.pt](https://drive.google.com/file/d/1sAO2OJ8j1kza2zH8MerbD6hGxdKsoogB/view?usp=sharing)
 * Flowers ours: [flower_p2.pt](https://drive.google.com/file/d/1d6DDKAEu_iwNzxlBaVrETHBcc6oF5jYf/view?usp=sharing)
 * MetFaces baseline: [metface_baseline.pt](https://drive.google.com/file/d/1SaHqew52S9iRCeN7kpPMLqlo2t34ekTb/view?usp=sharing)
 * MetFaces ours: [metface_p2.pt](https://drive.google.com/file/d/1swjgSB1WFF9JnBR6W6Newnfzdyo1nPYf/view?usp=sharing)
 
## Requirements
We tested on PyTorch 1.7.1, single RTX8000 GPU.

## Sampling from pre-trained models

First, set PYTHONPATH variable to point to the root of the repository. Do the same when training new models. 

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Put model checkpoints into a folder `models/`.

Samples will be saved in `samples/`.

```
python scripts/image_sample.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 250 --model_path models/ffhq_p2.pt --sample_dir samples
```

To sample for 250 timesteps without DDIM, replace `--timestep_respacing ddim25` to `--timestep_respacing 250`, and replace `--use_ddim True` with `--use_ddim False`.

## Training your models

`--p2_gamma` and `--p2_k` are two hyperparameters of P2 weighting. We used `--p2_gamma 0.5 --p2_k 1` and `--p2_gamma 1 --p2_k 1` in the paper.

Logs and models will be saved in `logs/`. You should modify `--data_dir`. 

We used lightweight version (93M parameter) of [ADM](https://arxiv.org/abs/2105.05233) (over 500M) as default model configuration. You may modify the model.

```
python scripts/image_train.py --data_dir data/DATASET_NAME --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 8 --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir logs
```


