# P2 weighting (CVPR 2022)

This is the codebase for [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227).

This repository is heavily based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

P2 modifies the weighting scheme of the training objective function to improve sample quality. It encourages the diffusion model to focus on recovering signals from highly corrupted data, where the model learns global and perceptually rich concepts. Below figure shows the weighting schemes in terms of SNR.

![snr_weight](https://user-images.githubusercontent.com/36615789/161203299-8b02d76b-9c51-4529-8329-3ac08e9f3bc8.png)

## Pre-trained models

All models are trained at 256x256 resolution.

Here are the models trained on FFHQ, CelebA-HQ, CUB, AFHQ-Dogs, Flowers, and MetFaces: [onedrive](https://1drv.ms/f/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Us79E9)  [gdrive](https://drive.google.com/drive/folders/1bcWh3XuQzdct4-UPTrIX-lvs47OiLaOM?usp=sharing)
 
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


