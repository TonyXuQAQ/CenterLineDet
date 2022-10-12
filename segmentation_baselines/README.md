# Baselines

In this part, we provide the implementation of baselines, i.e., HDMapNet ([code](https://github.com/Tsinghua-MARS-Lab/HDMapNet),[paper](https://arxiv.org/abs/2107.06307)) and our proposed FusionNet which is enhanced from HDMapNet. All baseline models are evaluated both in single-frame and multi-frame. For TopoRoad ([code](https://github.com/ybarancan/TopologicalLaneGraph),[paper](https://arxiv.org/pdf/2112.10155.pdf)), please check the original implementation of authors.


## HDMapNet
### Training
Run 
```
./bash/HD_train.bash
```
. Output and visualizations are saved in ```./HDMapNet```. The network is trained on the whole training set for 50 epochs.
### Inference
Run 
```
./bash/HD_inference.bash
```
. Output and visualizations are saved in ```./HDMapNet/inference```. 
### Training for CenterLineDet (Optional, if you want to train CenterLineDet-with-HDMapNet)
Since we need to train the CenterLineDet with training samples that are not seen by HDMapNet, in this section, we train HDMapNet on half of the training set for 50 epochs. The other half is used to train CenterLineDet-with-HDMapNet. 
Run 
```
./HD_for_CenterLineDet_train.bash
```
. Output and visualizations are saved in ```./HDMapNet_for_CenterLineDet```.


## FusionNet
### Training
Run 
```
./bash/Fusion_train.bash
```
. Output and visualizations are saved in ```./FusionNet```. The network is trained on the whole training set for 50 epochs.
### Inference
Run 
```
./bash/Fusion_inference.bash
```
. Output and visualizations are saved in ```./FusionNet/inference```. 
### Training for CenterLineDet (Optional, if you want to train CenterLineDet-with-FusionNet)
Since we need to train the CenterLineDet with training samples that are not seen by FusionNet, in this section, we train FusionNet on half of the training set for 50 epochs. The other half is used to train CenterLineDet-with-FusionNet. 
Run 
```
./Fusion_for_CenterLineDet_train.bash
```
. Output and visualizations are saved in ```./FusionNet_for_CenterLineDet```.



## Evaluation
Go to ```./eval```.