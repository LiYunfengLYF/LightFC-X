# LightFC-X: Lightweight Convolutional Tracker for RGB-X Tracking 

## Notice:
- Here rgbter-light is the initial name of our tracker, and rgbt-light is the initial name of our project



## Environment Installation

prepare your environment as [TBSI](https://github.com/RyanHTR/TBSI).

Notice: Our use pytorch version is 1.13.0

## Project Paths Setup
You can also modify paths by editing these two files
```
lib/test/evaluation/local.py  # paths about testing
```
env_num is used to distinguish different development devices, we recommend setting env_num to only 0 here




|  |  |  | RGBD | RGBD | RGBD | RGBT | RGBT | RGBT | RGBE | RGBE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Params | FLOPs | F | Fr | Re | PR | NPR | SR | PR | SR |
| LightFC-X-xxs <br/>(MobileFormer) | 3.83M | 0.79G | 0.408 | 0.407 | 0.410  | 54.8 | 51.5 | 44.7 | 64.4 | 49.8 |
| LightFC-X-xs<br/> (MobileViT) | 4.19M | 3.03G | 0.469 | 0.472 | 0.465 | 60.9 | 57.1 | 48.4 | 65.5 | 51.0  |
| LightFC-X | 6.68M | 5.40G | 0.500  | 0.537 | 0.468 | 63.8 | 59.9 | 50.1 | 68.7 | 52.9 |

RGBD is DepTrack. RGBT is LasHeR. RGBE is VisEvent.

Download LightFC-X-xs Model: https://drive.google.com/drive/folders/1EKxYogu-3WmTMTBASR6BtU2RQN90WhQp?usp=sharing

Download LightFC-X-xxs Model: https://drive.google.com/drive/folders/1xevE9iynAczPhkDs-VzqNgcGhbWEVQq5?usp=sharing

Download LightFC-X Model and results: https://drive.google.com/file/d/1rmhq42s_2ZaWZuuyihuJCUMBHM7anStG/view?usp=drive_link




## Data Preparation

Put LasHeR, RGBT234, VisEvent datasets' files into 

```
lib/test/evaluation/local.py  # paths about testing
```

Put DepthTrack, VOT22RGBD datasets into

```
Depthtrack_workspace/sequence
VOT22RGBD_workspace/sequences
```


## Training
Training method is same as OSTrack-like Library.

```
python tracking/train.py --script rgbter_light --config baselinev2_ep45_maevitt --save_dir ./output --mode multiple --env_num 3 --nproc_per_node 2 --use_wandb 0
```



## Test

Our checkpoints are released in 
```
${PROJECT_ROOT}
  -- output
      -- checkpoints
          -- lightfcx
              |-- RGBD_baseline_N3_ep45
              |-- RGBE_baseline_ep45
              |-- RGBS_baseline_ep30
              |-- RGBT_baseline_ep45
              |-- RGBT_baseline_update_ep15
```

Using the following cmds for eval our trackers, you have to make env_num match the env_num in lib/test/evaluation/local.py


### RGB-T trackers
```
# S
python tracking/test.py --script lightfcx --config RGBT_baseline_ep45  --dataset lasher --debug 0 --threads 4 --num_gpus 2 --env_num 0
python tracking/test.py --script lightfcx --config RGBT_baseline_ep45  --dataset rgbt234 --debug 0 --threads 4 --num_gpus 2 --env_num 0

# ST
python tracking/test.py --script lightfcx --config RGBT_baseline_update_ep15  --dataset lasher --debug 0 --threads 4 --num_gpus 2 --env_num 0
python tracking/test.py --script lightfcx --config RGBT_baseline_update_ep15  --dataset rgbt234 --debug 0 --threads 4 --num_gpus 2 --env_num 0

```

### RGB-E trackers
```
python tracking/test_rgbe.py --script_name rgbt_light --yaml_name RGBE_baseline_ep45
```

### RGB-D trackers

DepthTrack
```
cd Depthtrack_workspace
vot evaluate --workspace ./ rgbter_light
vot analysis --nocache --name rgbter_light
```
VOT22RGBD
```
cd VOT22RGBD_workspace
vot evaluate --workspace ./ rgbter_light
vot analysis --nocache --name rgbter_light
```
### RGB-S trackers

```
python tracking/test_rgbs.py
```


## Evaluation Profile Model

Evaluate RGB-X parameters and Macs:

```
python tracking/profile_model.py
```



## Evaluation Results
### RGB-T and RGB-E

Our RGB-T and RGB-E raw results are released in 
```
${PROJECT_ROOT}
  -- output
      -- test
          -- tracking_results
              -- lightfcx
                  |-- RGBE_baseline_ep45
                  |-- RGBE_baseline_update_ep15
                  |-- RGBT_baseline_ep45
                  |-- RGBT_baseline_update_ep15
```

Evaluate RGB-T results
```
python tracking/analysis_results.py --tracker_name lightfcx --tracker_param RGBT_baseline_ep45 --dataset lasher --env_num 2 
python tracking/analysis_results.py --tracker_name lightfcx --tracker_param RGBT_baseline_update_ep15 --dataset lasher --env_num 2 

```

Evaluate RGB-E results
```
python tracking/analysis_results.py --tracker_name lightfcx --tracker_param RGBE_baseline_ep45 --dataset lasher --env_num 2 
python tracking/analysis_results.py --tracker_name lightfcx --tracker_param RGBE_baseline_update_ep15 --dataset lasher --env_num 2 

```

### RGB-D
Our RGB-D raw results are released in


Analysis of DepthTrack
```
Depthtrack_workspace/analysis_RGBD_baseline_ep45
Depthtrack_workspace/analysis_RGBD_baseline_N3_ep45
Depthtrack_workspace/analysis_RGBD_baseline_N3_update_ep15
```


Analysis of VOT22RGBD
```
VOT22RGBD_workspace/analysis_RGBD_baseline_ep45
VOT22RGBD_workspace/analysis_RGBD_baseline_N3_ep45
VOT22RGBD_workspace/analysis_RGBD_baseline_N3_update_ep15
```

### RGB-S
Our RGB-D raw results are released in

```
${PROJECT_ROOT}
  -- output
      -- RGBPS
```

Evaluate RGB-S results
```
python tracking/analysis_rgbs.py
```

## Acknowledgment
