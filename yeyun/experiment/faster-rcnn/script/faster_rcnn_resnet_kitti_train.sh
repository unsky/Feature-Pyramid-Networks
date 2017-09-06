#!/usr/bin/env bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

train_coco_root_dir='/home/ubuntu/Work/brbchen/unskychen/dataset/coco'
val_coco_root_dir='/home/ubuntu/Work/brbchen/unskychen/dataset/coco'
train_voc_root_dir='/data/yeyun/data/VOCdevkit+/data/yeyun/data/VOCdevkit'
val_voc_root_dir='/data/yeyun/data/VOCdevkit'

train_kitti_root_dir='/home/ubuntu/Work/brbchen/unskychen/dataset/kitti'
val_kitti_root_dir='/home/ubuntu/Work/brbchen/unskychen/dataset/kitti'


output_folder='roi_concat-fpn_resnet-50-_8cls_4rpn'

if [ ! -d "${output_folder}" ]; then
mkdir "${output_folder}"
mkdir "./${output_folder}/model"
fi

network='--network=fpn_resnet-50'

dataset='--dataset=kitti'
train_image_sets='--image_sets=train'
train_img_root_dir="--dataset_root_path=${train_kitti_root_dir}"
val_image_sets='--val_image_sets=train'
val_img_root_dir="--valset_root_path=${val_kitti_root_dir}"

num_epoch='--num_epoch=20'
base_lr='--base_lr=0.001'
lr_step='--lr_step=7,13,17'

image_per_batch='--image_batch_size=1'
frequent='--frequent=100'
log_file="--log_file=./${output_folder}/${output_folder}.log"
save_prefix="--save_prefix=./${output_folder}/model/${output_folder}"

folder="--output_folder=${output_folder}"
gpus="--gpu='0,1'"

cmd="python train_end2end.py ${network} ${dataset} ${train_image_sets} ${train_img_root_dir} \
${val_image_sets} ${val_img_root_dir} ${num_epoch} ${base_lr} ${lr_step} ${image_per_batch} ${log_file} \
${save_prefix} ${frequent} ${folder} --gpus=0,1"

echo ${cmd}
${cmd} 2>&1 | tee -a ${output_folder}/cmd.log
