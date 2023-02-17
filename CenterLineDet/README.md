# Baselines

In this part, we provide the implementation of CenterLineDet with either HDMapNet or FusionNet as the perspective transformation network.

## Raw output
Find the raw output and evalution results in the following link [raw results](https://drive.google.com/file/d/1T6ZVuh_tmk6kQG38O6_Z4KFwhwZtiqqo/view?usp=share_link).

## CenterLineDet-with-HDMapNet
### Prerequisite
Make sure you have trained both HDMapNet and HDMapNet-for-centerlinedet. This step is necessary for both training and inference.

Then, run
```
./bash/HD_fuse_segmentation.bash
```
to generate segmentation maps.


### Sampling 
Run
```
./bash/HD_sample.bash
```
to generate training samples.

### Training 
Run
```
./bash/HD_train.bash
```
to train the network.

### Inference
Run 
```
./bash/HD_inference.bash
```
. Output and visualizations are saved in ```./HDMapNet/inference```. 

## CenterLineDet-with-FusionNet
### Prerequisite
Make sure you have trained both FusionNet and FusionNet-for-centerlinedet. This step is necessary for both training and inference.

Then, run
```
./bash/Fusion_fuse_segmentation.bash
```
to generate segmentation maps.


### Sampling 
Run
```
./bash/Fusion_sample.bash
```
to generate training samples.

### Training 
Run
```
./bash/Fusion_train.bash
```
to train the network.

### Inference
Run 
```
./bash/Fusion_inference.bash
```
. Output and visualizations are saved in ```./FusionMapNet/inference```. 


## Evaluation
Go to ```./eval```.