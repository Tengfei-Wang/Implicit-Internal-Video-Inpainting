# Implicit Internal Video Inpainting
Implementation for our ICCV2021 paper 'Internal Video Inpainting by Implicit Long-range Propagation'


[paper]( ) | [project website](https://tengfei-wang.github.io/Implicit-Internal-Video-Inpainting/) | [4K data](https://tengfei-wang.github.io/Implicit-Internal-Video-Inpainting/) | [video](https://youtu.be/VlDSJtmBqBs)

## Introduction
We proposed a  simple but effective video inpainting method. The inpainting process is zero-shot and implicit, which does not need any pretraining on large video datasets or optical-flow estimation. We further extend the proposed method to more challenging tasks:  video object removal with  limited annotated  masks,  and inpainting on ultra high-resolution videos (e.g., 4K videos).
<img src="pics/demo.jpg" height="550px"/> 

### TO DO
- [x] Release base code
- [ ] Release code fot mask prediction
- [ ] Release code and data fot 4K videos


## Setup

### Environment
```
conda create -n IIVI python=3.7
conda activate IIVI
conda install tensorflow-gpu tensorboard
pip install pyaml 
```
The environment can also be set up by the provided environment.yml.


### Installation
```
git clone https://github.com/Tengfei-Wang/Implicit-Internal-Video-Inpainting.git
cd Implicit-Internal-Video-Inpainting
```

## Quick Start 
To try our  method:
```

```
The  results are placed in ./exp/results.
 
## Try Your Own Data
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
