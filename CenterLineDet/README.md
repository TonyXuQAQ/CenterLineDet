# CenterLineDet

In this part, we provide the implementation of CenterLineDet with either HDMapNet or FusionNet as the perspective transformation network.


## CenterLineDet-with-HDMapNet
### Prerequisite
Make sure you have trained both HDMapNet and HDMapNet-for-centerlinedet. This step is necessary for both training and inference.

Then, run
```
./bash/HD_fuse_segmentation.bash
```
to generate segmentation maps.


### Sampling and Training
Commming soon...

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


### Sampling and Training
Commming soon...

### Inference
Run 
```
./bash/Fusion_inference.bash
```
. Output and visualizations are saved in ```./FusionMapNet/inference```. 


## Evaluation
Go to ```./eval```.