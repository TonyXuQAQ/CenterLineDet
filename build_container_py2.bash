#!/bin/bash

CMD=$*

if [ -z "$CMD"];
then 
	CMD=/bin/bash
fi

home_dir=/home/zhxu/final_repo/CenterLineDet # the dir of CenterLineDet
dataset_dir=/home/zhxu/nuscenes # the dir of nuscenes dataset
container_name=centerlinedet_eval # container name
port_number=5050 # port number for tensorboard

docker run -d \
	-v $home_dir:/tonyxu\
	-v $dataset_dir:/tonyxu/dataset\
	--name=$container_name\
	--gpus all\
	--shm-size 32G\
	-p $port_number:6006\
	--rm -it zhxu_py2 $CMD

docker attach centerlinedet_eval