

## Run evaluation
First make sure you run ```./build_container_py2.bash``` and you are in the container with python2 available. Then run 
```
eval.bash
```
Evaluation results are saved in ```../$model/inference/multi(or single)/results```

## Pixel-level metrics
Check ```./pixel_metrics.py``` for code. Pixel-level evaluation results are saved in ```../$model/inference/multi(or single)/results/pixel.json```


## TOPO (Topology-level metrics)
TOPO metrics are implemented by [SAT2GRAPH](https://github.com/songtaohe/Sat2Graph). We modify some parameters for our evaluation. Topo-level evaluation results are saved in ```../$model/inference/multi(or single)/results/topo.json```

### TOPO Usage
```bash
python main.py -graph_gt example/gt.p -graph_prop example/prop.p -output toporesult.txt
```

Here, the graph files gt.p and prop.p are all in the same format as what we used in Sat2Graph - a python dictionary where each key is the coordinate of a vertex (denoted by x) and the corresponding value is a list of x's neighboring vertices.  


### TOPO Parameters
Parameters | Note 
--------------------- | -------------
Propagation Distance  | 100 meters
Propagation Interval  | Default is 1 meters. Config with -interval flag.
Matching Distance Threshold | Default is 2 meters. Config with -matching_threshold flag.
Matching Angle Threshold | 30 degrees 
One-to-One Matching | True


### Dependency
* hopcroftkarp
* rtree
