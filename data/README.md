## Data Preparation
1. Download Nuscenes in the same way as [HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet). In addition, you need to download the nuscenes map expansion.
2. Run 
    ```
    ./prepare_dataset.bash
    ```
    to generate training labels.

3. **Optional**. Data split info is recorded in ```./split.json```. The training set is further split into ```train_1``` containing 349 scenes and ```train_2``` containing 349 scenes. Both ```train_1``` and ```train_2``` are used if you want to train perspective transformation module only (i.e., HDMapNet or FusionNet). If you want to train CenterLineDet, first train the perspective transformation module by ```train_1```, and train CenterLineDet by ```train_2```. 

    There are 148 scenes for testing (which is the same as the data split of TopoRoad). For training efficiency, we only select 10 scenes from testing set as validation scenes to monitors the training process.

4. **Optional**. Road centerline graphs of nuscenes are provided in ```./maps```, which can be directly used for our task. But you can still re-create centerline graphs by running
    ```
    python get_map_json_label.py
    ```
    , which could be helpful if you want to adapt our approaches to your customized dataset.


## Download pretrained checkpoints
Run 
```
./get_pretrained_checkpoints.bash
```
