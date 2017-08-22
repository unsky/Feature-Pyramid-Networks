Feature Pyramid Network on caffe

This is the unoffical version  Feature Pyramid Network for Feature Pyramid Networks for Object Detection https://arxiv.org/abs/1612.03144


the caffe unoffical version  Feature Pyramid Network: https://github.com/unsky/FPN

# usage

1. install mxnet v0.9.5

2. dowload resnet_v1_101-0000.params and your dataset

3. init
```
./init.sh
```
### train
python train.py --cfg coco.yaml 

### test
python test.py --cfg coco.yaml 


if you have issue about the fpn, open an issue.
