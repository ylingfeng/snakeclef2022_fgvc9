# 4th Place Solution for FGVC9 with PyTorch Implementation

## Device
ViT-large on 8 x Titan XP (12GB)<br>
ViT-huge on 8 x 3090 ti (24GB)<br>
<br>

## Requirements
- python 3.7
- pytorch 1.7.1+cu101
- torchvision 0.8.2
- timm 0.3.2

## Preparation
1.&ensp;Get SnakeCLEF 2022 dataset<br>

&emsp;&emsp;&emsp;root/<br>
&emsp;&emsp;&emsp;&emsp;├─ SnakeCLEF2022-ISOxSpeciesMapping.csv<br>
&emsp;&emsp;&emsp;&emsp;├─ train/<br>
&emsp;&emsp;&emsp;&emsp;│&emsp;&emsp;├─ SnakeCLEF2022-TrainMetadata.csv<br>
&emsp;&emsp;&emsp;&emsp;│&emsp;&emsp;├─ SnakeCLEF2022-small_size/<br>
&emsp;&emsp;&emsp;&emsp;│&emsp;&emsp;├─ SnakeCLEF2022-medium_size/<br>
&emsp;&emsp;&emsp;&emsp;│&emsp;&emsp;└─ SnakeCLEF2022-large_size/<br>
&emsp;&emsp;&emsp;&emsp;└─ test/<br>
&emsp;&emsp;&emsp;&emsp; &ensp;&emsp;&emsp;├─ SnakeCLEF2022-TestMetadata.csv<br>
&emsp;&emsp;&emsp;&emsp; &ensp;&emsp;&emsp;└─ SnakeCLEF2022-large_size/<br>

2.&ensp;Get MAE pretrained models following [README_MAE.md](./README_MAE.md)
   
3.&ensp;Calculate sample per class
```cmd
python preprocess_sample_per_class.py
```
output: ```./preprocessing/sample_per_class.json```<br><br>
4.&ensp;Prepocess metadata
```cmd
python preprocess_endemic_metadata.py
```
output: ```./preprocessing/endemic_label.json```<br>
```cmd
python preprocess_code_metadata.py
```
output: ```./preprocessing/code_label_train.json```<br>
&emsp;&emsp;&emsp;&ensp;```./preprocessing/code_label_test.json```

<br>

## ViT-large
### Train
```cmd
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
  --accum_iter 4 \
  --batch_size 2 \
  --input_size 432 \
  --model vit_large_patch16 \
  --epochs 50 \
  --blr 1e-3 \
  --layer_decay 0.75 \
  --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --root root/to/your/data \
  --data snakeclef2022 \
  --nb_classes 1572 \
  --log_dir ./log_dir/vit_large_patch16_432_50e \
  --output_dir ./output_dir/vit_large_patch16_432_50e \
  --finetune ./pretrained_model/mae_pretrain_vit_large.pth \
  --use_prior --loss LogitAdjustment
```
### Test and genrate submission file
```cmd
python main_finetune.py \
  --accum_iter 4 \
  --batch_size 64 \
  --input_size 432 \
  --model vit_large_patch16 \
  --epochs 50 \
  --blr 1e-3 \
  --layer_decay 0.75 \
  --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --root root/to/your/data \
  --data snakeclef2022 \
  --nb_classes 1572 \
  --log_dir ./log_dir/vit_large_patch16_432_50e \
  --output_dir ./output_dir/vit_large_patch16_432_50e \
  --resume ./output_dir/vit_large_patch16_432_50e/checkpoint-xx.pth \
  --use_prior --loss LogitAdjustment \
  --eval --test \
  --tencrop --crop_pct 0.875
```



## ViT-huge
### Train
```cmd
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
  --accum_iter 4 \
  --batch_size 2 \
  --input_size 392 \
  --model vit_huge_patch14 \
  --epochs 45 \
  --blr 1e-3 \
  --layer_decay 0.8 \
  --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --root root/to/your/data \
  --data snakeclef2022 \
  --nb_classes 1572 \
  --log_dir ./log_dir/vit_huge_patch14_392_40e \
  --output_dir ./output_dir/vit_huge_patch14_392_40e \
  --finetune ./pretrained_model/mae_pretrain_vit_huge.pth \
  --use_prior --loss LogitAdjustment
```
### Test and genrate submission file
```cmd
python main_finetune.py \
  --accum_iter 4 \
  --batch_size 64 \
  --input_size 392 \
  --model vit_huge_patch14 \
  --epochs 45 \
  --blr 1e-3 \
  --layer_decay 0.8 \
  --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --root root/to/your/data \
  --data snakeclef2022 \
  --nb_classes 1572 \
  --log_dir ./log_dir/vit_huge_patch14_392_40e \
  --output_dir ./output_dir/vit_huge_patch14_392_40e \
  --resume /data/mae/output_dir/vit_huge_patch14_392_40e/checkpoint-xx.pth \
  --use_prior --loss LogitAdjustment \
  --eval --test \
  --tencrop --crop_pct 0.875
```
<br>

## Ensemble
```cmd
python ensemble.py
```

## Results

| model     | resolution | public  | private |                                          checkpoint                                          |
| --------- | :--------: | :-----: | :-----: | :------------------------------------------------------------------------------------------: |
| ViT-large |    384     | 0.87996 | 0.81997 | [Google](https://drive.google.com/file/d/1Rpax1cS5uE5rGYa2nuUyLdhZ1SuZa0pf/view?usp=sharing) |
| ViT-large |    432     | 0.89173 | 0.83063 | [Google](https://drive.google.com/file/d/1vnNqoCa9723XgZ7Izw48VqppXMILPqz7/view?usp=sharing) |
| ViT-huge  |    392     | 0.89449 | 0.84057 | [Google](https://drive.google.com/file/d/1EEd7KllY2Z0gvLzZaLyD0VLMHS4Fc0DH/view?usp=sharing) |
| Ensemble  |     --     | 0.89822 | 0.84565 |                                              --                                              |
