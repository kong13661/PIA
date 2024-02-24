# PIA
Official repo for An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization

## Requirements

Follow `gradtts/train/README.md`.

## DDPM
### train DDPM model

We provide the split of the dataset. They are `DDPM/CIFAR10_train_ratio0.5.npz` and `DDPM/TINY-IN_train_ratio0.5.npz`. To train the DDPM, you need put the `cifar10` dataset into `DDPM/data/pytorch` and Tiny-ImageNet to `DDPM/data/tiny-imagenet-200`. You can also change the directory by modifying the path in `main.get_dataset` function and `dataset_utils.load_member_data`.  You can change the log directory by modifying `FLAGS.logdir` in `main.py`. You can change the `FLAGS.dataset` to select the dataset.

Then, to train the DDPM, just run command below.
```bash
cd DDPM
python main.py
```

### attack DDPM model
Just run command below.
```bash
cd DDPM
python attack.py --checkpoint your_checkpoint --dataset your_dataset --attacker_name attacker_name --attack_num attack_num --interval interval
```

The meaning of those parameters:

`--checkpoint` The checkpoint you saved.

`--dataset` The dataset to attack. It can be `cifar10` or `TINY-IN`.

`--attacker_name` The attack method. `naive` for NA in our paper. `SecMI` for SceMI attack. `PIA` for PIA and `PIAN` for PIAN

`--attack_num` attack number from $t=0$

`--interval` attack interval. For example, if `attack_num=5`, `interval=20`,  the attack method will attack \[20, 40, 60, 80, 100\]. 

At last, this program will print AUC and TPR @ 1% FPR in \[20, 40, 60, 80, 100\].

### apply our method to other model

Inherit a subclass from `components.EpsGetter`, implement `__call__` method, and return predicted $\epsilon$. `noise_level[t]` is $\prod_{k=1}^{t} (1-\beta_k)$. Keep other same with `attack.py`.

## GradTTS
### train GradTTS model
Code in `gradtts/train` is the official code of gradtts. We provide the dataset split in `gradtts/train/split`.  Just put [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) and [LibriTTS](https://www.openslr.org/60/) into `gradtts/train/datasets`. For LibriTTS, to use our pretrained checkpoint, your need to resample audio to 22050Hz. Run command below to resample:

```bash
cd gradtts/train
python resample.py
```

To train the model, your can chagne parameters in `gradtts/train/params.py` for ljspeech and `gradtts/train/params_libritts.py` for libritts, especially for `train_filelist_path`, `valid_filelist_path` and `log_dir`. By default, without any change, code can work properly.

```bash
cd gradtts/train
python train.py   # for ljspeech
python train_multi_speaker_libritts.py  # for libritts
```

### attack Stable Diffusion

Just run command below.
```bash
cd DDPM
python attack.py --checkpoint your_checkpoint --dataset your_dataset --attacker_name attacker_name --attack_num attack_num --interval interval
```

The meaning of those parameters:

`--checkpoint` The checkpoint you saved.

`--dataset` The dataset to attack. It can be `laion5`, `laion5_none` (no groundtruth text) or `laion5_blip` (blip generated text).

`--attacker_name` The attack method. `naive` for NA in our paper. `SecMI` for SceMI attack. `PIA` for PIA and `PIAN` for PIAN

`--attack_num` attack number from $t=0$

`--interval` attack interval. For example, if `attack_num=5`, `interval=20`,  the attack method will attack \[20, 40, 60, 80, 100\]. 

At last, this program will print AUC and TPR @ 1% FPR in \[20, 40, 60, 80, 100\].

You to attack stable diffusion, you need to install `diffusers==0.18.0`.

We also provide the images evaluated in our paper. Download from [MIA_efficient](https://1drv.ms/f/s!Aqkz6X6nVZGDiY4TMwyHSP2Ij-rinA?e=BldxvP). If you download the data, you could modify `/home/kongfei/workspace/PIA/stable_diffusion/dataset.py` and set up `coco_dataset_root`, `coco_dataset_anno` and `stable_diffusion_data`. The you also need to download COCO dataset by yourself.

### attack GradTTS

Change the value of `train_filelist_path`, `valid_filelist_path`  in `gradtts/attack/gradtts/params_ljspeech.py` and `gradtts/attack/gradtts/params_libritts.py` to load dataset.

```bash
cd gradtts/attack
python attack.py --checkpoint your_checkpoint --dataset your_dataset --attacker_name attacker_name --attack_num attack_num --interval interval
```

The meaning of parameters is same with that in DDPM.

`--dataset` You can use `ljspeech` or `libritts`.


## Pretrained Checkpoint
You can download pretrained checkpoint from [MIA_efficient](https://1drv.ms/f/s!Aqkz6X6nVZGDiY4TMwyHSP2Ij-rinA?e=BldxvP).