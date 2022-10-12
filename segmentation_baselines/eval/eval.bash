model=FusionNet


# pixel level
python3 pixel_metric.py --root_dir $model/inference/multi --name multi
python3 pixel_metric.py --root_dir $model/inference/single --name single

# topo level
python main.py -savedir $model/inference/multi
python post_process.py -savedir $model/inference/multi

