import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

from PIL import Image, ImageDraw

import json

class Lane():
    def __init__(self,id,start,end,vertices):
        self.id = id
        self.start = start
        self.end = end
        self.vertices = vertices
        self.lane_neighbors = []

    def reverse(self):
        temp = self.end
        self.end = self.start
        self.start = temp
        self.vertices = self.vertices[::-1]

class Junction():
    def __init__(self,id,node):
        self.id = id
        self.x = node[0]
        self.y = node[1]
        self.z = node[2]
        self.dead = False
        self.junc_neighbors = []

class Graph():
    def __init__(self):
        self.lanes = {}
        self.junctions = {}
        self.lane_num = 0
        self.junction_num = 0

    def add_lane(self,data):
        lane = Lane(self.lane_num,data['start_node'],data['end_node'],data['vertices'])
        self.lanes[self.lane_num] = lane
        self.lane_num += 1

    def add_junction(self,v):
        junction = Junction(self.junction_num,v)
        self.junctions[self.junction_num] = junction
        self.junction_num += 1

    def find_junction(self,v):
        for _,j in self.junctions.items():
            if j.x==v[0] and j.y==v[1] and j.z==v[2]:
                return j
        return None
    
    def merge_lane(self,l1,l2):
        if l1.end==l2.start:
            pass
        elif l1.end==l2.end:
            l2.reverse()
        elif l1.start==l2.end:
            l1.reverse()
            l2.reverse()
        elif l1.start==l2.start:
            l1.reverse()
        else:
            raise "Error"
        merged_lane = Lane(self.lane_num,l1.start,l2.end,l1.vertices+l2.vertices)
        self.lanes[self.lane_num] = merged_lane
        self.lane_num += 1
        return merged_lane

def get_orientation_angle(vector):
    norm = np.linalg.norm(vector)
    theta = 0
    if norm:
        vector = vector / norm
        theta = np.arccos(vector[0])
        if vector[1] > 0:
            theta = 2*np.pi - theta
    return theta * 180 / np.pi

MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
for map_name in MAPS:
    nusc_map = NuScenesMap(dataroot='./dataset', map_name=map_name)
    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
    graph_list = nusc_map.get_graph(resolution_meters=1)

    # with open(f'./{nusc_map.map_name}_points.json','w') as jf:
    #     json.dump(graph_list,jf)

    graph = Graph()
    for data in graph_list:
        graph.add_lane(data)
        # process the start node
        if graph.find_junction(data['start_node']) is None:
            graph.add_junction(data['start_node'])
            
        # process the end node
        if graph.find_junction(data['end_node']) is None:
            graph.add_junction(data['end_node'])

    # connection
    for _, lane in graph.lanes.items():
        junction = graph.find_junction(lane.start)
        if junction is not None:
            lane.lane_neighbors.append(junction)
            junction.junc_neighbors.append(lane)
        junction = graph.find_junction(lane.end)
        if junction is not None:
            lane.lane_neighbors.append(junction)
            junction.junc_neighbors.append(lane)


    for _, junction in graph.junctions.items():
        if len(junction.junc_neighbors) == 2:
            lane1 = junction.junc_neighbors[0]
            lane2 = junction.junc_neighbors[1]
            graph.lanes.pop(lane1.id)
            graph.lanes.pop(lane2.id)
            junction.dead = True
            lane1.lane_neighbors.remove(junction)
            lane2.lane_neighbors.remove(junction)
            junction1 = lane1.lane_neighbors[0]
            junction2 = lane2.lane_neighbors[0]
            #
            
            merged_lane = graph.merge_lane(lane1,lane2)
            merged_lane.lane_neighbors.append(junction1)
            merged_lane.lane_neighbors.append(junction2)
            junction1.junc_neighbors.remove(lane1)
            junction1.junc_neighbors.append(merged_lane)
            junction2.junc_neighbors.remove(lane2)
            junction2.junc_neighbors.append(merged_lane)


    output_list = {'lanes':[],'junctions':[]}
    for _, lane in graph.lanes.items():
        if len([x for x in lane.lane_neighbors if not x.dead]) and len(lane.vertices)>10:
            orientation = []
            for i in range(1,len(lane.vertices)):
                vector = np.array(lane.vertices[i]) - np.array(lane.vertices[i-1])
                orientation.append(get_orientation_angle(vector))
            orientation.append(get_orientation_angle(vector))
            output_list['lanes'].append({'id':lane.id,'vertices':[[round(x[0],4),round(x[1],4)] for x in lane.vertices],'orientation':[round(x,4) for x in orientation]})

    for _, junction in graph.junctions.items():
        if junction.dead==False:
            output_list['junctions'].append({'id':junction.id,'x':round(junction.x,4),'y':round(junction.y,4),'neighbors':[n.id for n in junction.junc_neighbors]})
        

    with open(f'./maps/{nusc_map.map_name}.json','w') as jf:
        json.dump(output_list,jf)