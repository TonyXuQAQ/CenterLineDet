import numpy as np
import os
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('-savedir', type=str)
args = parser.parse_args()

topo = []
precision = []
recall = []
output = []
file_list = os.listdir('../%s/results/topo'%args.savedir)
file_list.sort()
for file_name in file_list:
    if '.txt' not in file_name:
        continue
    with open('../%s/results/topo/%s'%(args.savedir,file_name)) as f:
        lines = f.readlines()
    p = float(lines[-2].split(' ')[0].split('=')[-1])
    r = float(lines[-2].split(' ')[-1].split('=')[-1])
    f1 = float(lines[-1].split(':')[-1])
    topo.append(f1)
    precision.append(p)
    recall.append(r)
    # print(file_name,topo[-1])
    output.append({'id':file_name,'p/r/f1':[p,r,f1]})

print('Precision',np.mean(precision),'Recall',np.mean(recall),'TOPO',np.mean(topo))
with open('../%s/results/topo.json'%args.savedir,'w') as jf:
    json.dump({'mean topo':[np.mean(precision),np.mean(recall),np.mean(topo)],'detail':output},jf)