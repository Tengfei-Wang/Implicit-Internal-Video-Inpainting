# Implicit Internal Video Inpainting
Implementation for our ICCV2021 paper 'Internal Video Inpainting by Implicit Long-range Propagation'


[paper]( ) | [project website](https://tengfei-wang.github.io/Implicit-Internal-Video-Inpainting/) | [4K data](https://tengfei-wang.github.io/Implicit-Internal-Video-Inpainting/) | [demo video](https://youtu.be/VlDSJtmBqBs)

<img src="pics/boxing-fisheye-input.gif" width="180px"/>    <img src="pics/shooting-input.gif" width="180px"/>   <img src="pics/horsejump-high-input.gif" width="180px"/>    <img src="pics/gold-fish-input.gif" width="180px"/> 

<img src="pics/boxing-fisheye.gif" width="180px"/>    <img src="pics/shooting.gif" width="180px"/>   <img src="pics/horsejump-high.gif" width="180px"/>    <img src="pics/gold-fish.gif" width="180px"/> 


## Introduction
We proposed a  simple but effective video inpainting method. The inpainting process is zero-shot and implicit, which does not need any pretraining on large video datasets or optical-flow estimation. We further extend the proposed method to more challenging tasks:  video object removal with  limited annotated  masks,  and inpainting on ultra high-resolution videos (e.g., 4K videos).
<img src="pics/demo.jpg" height="500px"/> 

### TO DO
- [x] Release base code
- [x] Release code for mask propagation
- [ ] Release code and data for 4K videos


## Setup

### Environment
This code is based on tensorflow 2.x  (tested on tensorflow 2.2, 2.4).

The environment can be simply set up by Anaconda:
```
conda create -n IIVI python=3.7
conda activate IIVI
conda install tensorflow-gpu tensorboard
pip install pyaml 
pip install opencv-python
```

Or, you can also   set up the environment from the provided environment.yml:
```
conda env create -f environment.yml
```


### Installation
```
git clone https://github.com/Tengfei-Wang/Implicit-Internal-Video-Inpainting.git
cd Implicit-Internal-Video-Inpainting
```

## Usage
### Quick Start
We provide an example sequence 'bmx-trees'  in `./inputs/` . To try our  method:
```
python train.py
```
The default iterations is set to 40,000 in `config/train.yml`, and the internal learning takes ~3 hours with a single GPU. 
During the learning process, you can use tensorboard to check the inpainting results by:
```
tensorboard --logdir ./exp/logs
```
After the training, the final results can be saved in `./exp/results/` by:
```
python test.py
```
You can also modify  'model_restore' in `config/test.yml` to save results with different checkpoints.
 
### Try Your Own Data
#### Data preprocess
Before training, we advise to dilate the object masks first to exclude some edge pixels. Otherwise, the imperfectly-annotated masks would lead to artifacts in the object removal task.

You can generate and preprocess the mask by this script:
```
python scripts/preprocess_mask.py
```

#### Basic training
Modify the `config/train.yml`, which indicates the video path, log path, and training iterations,etc.. The training iterations depends on the video length, and it typically takes 30,000 ~ 80,000 iterations for convergence for 100-frame videos.
By default, we only use reconstruction loss for training, and it works well for most cases. 
```
python train.py
```

#### Improve the sharpness and consistency
For some hard videos, the former training may not produce a pleasing result. You can fine-tune the trained model with another losses.
To this end, modify the 'model_restore' in `config/test.yml` to the checkpoint path of basic training. Also set use_ambiguity_loss or use_stable_loss to True. Then fine-tune the 
basic checkpoint for 20,000-40,000 iterations.
```
python train.py
```

#### Inference
Modify the `config/test.yml`, which indicates the video path, log path, and save path.
```
python test.py
```

## Citation
If you find this work useful for your research, please cite:
``` 
@inproceedings{ouyang2021video,
  title={Internal Video Inpainting by Implicit Long-range Propagation},
  author={Ouyang, Hao and Wang, Tengfei and Chen, Qifeng},
  booktitle={International Conference on Computer Vision (ICCV) },
  year={2021}
} 
```


## Contact
Please send emails to [Hao Ouyang](ououkenneth@gmail.com) or [Tengfei Wang](tengfeiwang12@gmail.com)  if there is any question
