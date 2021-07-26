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
- [ ] Release code fot mask prediction
- [ ] Release code and data fot 4K videos


## Setup

### Environment
This code is based on tensorflow 2.x  (tested on tensorflow 2.2, 2.4).

The environment can be simply set up by Anaconda:
```
conda create -n IIVI python=3.7
conda activate IIVI
conda install tensorflow-gpu tensorboard
pip install pyaml 
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
To try our  method:
```

```
The  results are placed in ./exp/results.
 
### Try Your Own Data
To try our  method:
```

```
The  results are placed in ./exp/results.

## Citation
If you find this work useful for your research, please cite:
``` 
 
```


## Contact
Please send emails to [Hao Ouyang](ououkenneth@gmail.com) or [Tengfei Wang](tengfeiwang12@gmail.com)  if there is any question
