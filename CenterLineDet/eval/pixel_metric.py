import numpy as np
import json
import os
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse
from skimage.morphology import skeletonize


def calculate_scores(gt_points,pred_points):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0,0,0
    thr = 3
    dis_gt2pred,_ = pred_tree.query(gt_points, k=1)
    dis_pred2gt,_ = gt_tree.query(pred_points, k=1)
    recall = len([x for x in dis_gt2pred if x<thr])/len(dis_gt2pred)
    acc = len([x for x in dis_pred2gt if x<thr])/len(dis_pred2gt)
    r_f = 0
    if acc*recall:
        r_f = 2*recall * acc / (acc+recall)
    return acc, recall, r_f

def pixel_eval_metric(pred_mask,gt_mask):
    def tuple2list(t):
        return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

    gt_points = tuple2list(np.where(gt_mask!=0))
    pred_points = tuple2list(np.where(pred_mask!=0))

    return calculate_scores(gt_points,pred_points)

parser = argparse.ArgumentParser()
# logging config
parser.add_argument("--root_dir", type=str, default='./')
parser.add_argument('--name',type=str, default='./')

args = parser.parse_args()
os.makedirs(f'../{args.root_dir}/results',exist_ok=True)
image_list = os.listdir(os.path.join('../',args.root_dir,'pred_mask'))
image_list.sort()

with open('../../data/split.json','r') as jf:
    test_list = json.load(jf)['test']
image_list = [x for x in image_list if x[:10] in test_list]


pixel_scores = []
topo_scores = []
with tqdm(total=len(image_list),unit='img') as pbar:
    for ii, image_name in enumerate(image_list):
        pred_mask = np.array(Image.open(os.path.join('../',args.root_dir,'pred_mask',image_name)))
        if 'multi' in args.name:
            gt_mask = np.array(Image.open(os.path.join('../../dataset/label_gt_mask',image_name)))
        else:
            gt_mask = np.array(Image.open(os.path.join('../',args.root_dir,'gt_mask',image_name)))
        if len(pred_mask.shape)==3:
            pred_mask = pred_mask[:,:,0]
        if len(gt_mask.shape)==3:
            gt_mask = gt_mask[:,:,0]
        pred_mask = skeletonize(pred_mask, method='lee')
        pixel_scores.append(pixel_eval_metric(pred_mask,gt_mask))
        pbar.set_description(f'PP/PR/PF: {round(sum([x[0] for x in pixel_scores])/len(pixel_scores),3)}/{round(sum([x[1] for x in pixel_scores])/len(pixel_scores),3)}/{round(sum([x[2] for x in pixel_scores])/len(pixel_scores),3)}')
        pbar.update()
    
with open(os.path.join('../',args.root_dir,f'./results/pixel.json'),'w') as jf:
    json.dump({'PP':sum([x[0] for x in pixel_scores])/len(pixel_scores),
                'PR':sum([x[1] for x in pixel_scores])/len(pixel_scores),
                'PF':sum([x[2] for x in pixel_scores])/len(pixel_scores),
                'pixel_scores':[{f'{image_list[i]}':x} for i,x in enumerate(pixel_scores)]},jf)