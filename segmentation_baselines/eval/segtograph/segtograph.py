#import rdp
# Code Copied From Favyen
import sys 
import os 
sys.path.append(os.path.dirname(sys.path[0]))

import os
from PIL import Image
import math
import numpy
import numpy as np
from multiprocessing import Pool
import sys
from math import sqrt
import pickle
from .postprocessing import graph_refine, connectDeadEnds, downsample
import cv2 
from .douglasPeucker import simpilfyGraph
import argparse
import json





class Vertex():
    def __init__(self,v,id):
        self.x = v[0]
        self.y = v[1]
        self.id = id
        self.neighbors = []

class Edge():
    def __init__(self,src,dst,id):
        self.src = src
        self.dst = dst
        self.id = id

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.vertex_num = 0
        self.edge_num = 0

    def find_v(self,v_coord):
        if f'{v_coord[0]}_{v_coord[1]}' in self.vertices.keys():
            return self.vertices[f'{v_coord[0]}_{v_coord[1]}']
        return 

    def find_e(self,v1,v2):
        if f'{v1.id}_{v2.id}' in self.edges:
            return True
        return None

    def add(self,edge):
        v1_coord = edge[0]
        v2_coord = edge[1]
        v1 = self.find_v(v1_coord)
        if v1 is None:
            v1 = Vertex(v1_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v1.x}_{v1.y}'] = v1
        
        v2 = self.find_v(v2_coord)
        if v2 is None:
            v2 = Vertex(v2_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v2.x}_{v2.y}'] = v2

        if v1 not in v2.neighbors:
            v2.neighbors.append(v1)
        if v2 not in v1.neighbors:
            v1.neighbors.append(v2)
        e = self.find_e(v1,v2)
        if e is None:
            self.edges[f'{v1.id}_{v2.id}'] = Edge(v1,v2,self.edge_num)
            self.edge_num += 1
            self.edges[f'{v2.id}_{v1.id}'] = Edge(v2,v1,self.edge_num)
            self.edge_num += 1


def distance(a, b):
	return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
	if (start == end):
		return distance(point, start)
	else:
		n = abs(
			(end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
		)
		d = sqrt(
			(end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
		)
		return n / d

def rdp(points, epsilon):
	"""
	Reduces a series of points to a simplified version that loses detail, but
	maintains the general shape of the series.
	"""
	dmax = 0.0
	index = 0
	for i in range(1, len(points) - 1):
		d = point_line_distance(points[i], points[0], points[-1])
		if d > dmax:
			index = i
			dmax = d
	if dmax >= epsilon:
		results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
	else:
		results = [points[0], points[-1]]
	return results

def post_process_skeleton(im,output_dir,name,save_graph=False):
	if len(im.shape) == 3:
		im = im[:, :, 0]
	im = im.astype('uint8')

	vertices = []
	edges = set()
	def add_edge(src, dst):
		if (src, dst) in edges or (dst, src) in edges:
			return
		elif src == dst:
			return
		edges.add((src, dst))
	point_to_neighbors = {}
	q = []
	while True:
		if len(q) > 0:
			lastid, i, j = q.pop()
			path = [vertices[lastid], (i, j)]
			if im[i, j] == 0:
				continue
			point_to_neighbors[(i, j)].remove(lastid)
			if len(point_to_neighbors[(i, j)]) == 0:
				del point_to_neighbors[(i, j)]
		else:
			w = numpy.where(im > 0)
			if len(w[0]) == 0:
				break
			i, j = w[0][0], w[1][0]
			lastid = len(vertices)
			vertices.append((i, j))
			path = [(i, j)]

		while True:
			im[i, j] = 0
			neighbors = []
			for oi in [-1, 0, 1]:
				for oj in [-1, 0, 1]:
					ni = i + oi
					nj = j + oj
					if ni >= 0 and ni < im.shape[0] and nj >= 0 and nj < im.shape[1] and im[ni, nj] > 0:
						neighbors.append((ni, nj))
			if len(neighbors) == 1 and (i, j) not in point_to_neighbors:
				ni, nj = neighbors[0]
				path.append((ni, nj))
				i, j = ni, nj
			else:
				if len(path) > 1:
					path = rdp(path, 2)
					if len(path) > 2:
						for point in path[1:-1]:
							curid = len(vertices)
							vertices.append(point)
							add_edge(lastid, curid)
							lastid = curid
					neighbor_count = len(neighbors) + len(point_to_neighbors.get((i, j), []))
					if neighbor_count == 0 or neighbor_count >= 2:
						curid = len(vertices)
						vertices.append(path[-1])
						add_edge(lastid, curid)
						lastid = curid
				for ni, nj in neighbors:
					if (ni, nj) not in point_to_neighbors:
						point_to_neighbors[(ni, nj)] = set()
					point_to_neighbors[(ni, nj)].add(lastid)
					q.append((lastid, ni, nj))
				for neighborid in point_to_neighbors.get((i, j), []):
					add_edge(neighborid, lastid)
				break
	neighbors = {}


	vertex = vertices

	for edge in edges:

		nk1 = (vertex[edge[0]][1],vertex[edge[0]][0])
		nk2 = (vertex[edge[1]][1],vertex[edge[1]][0])
		
		if nk1 != nk2:
			if nk1 in neighbors:
				if nk2 in neighbors[nk1]:
					pass
				else:
					neighbors[nk1].append(nk2)
			else:
				neighbors[nk1] = [nk2]

			if  nk2 in neighbors:
				if nk1 in neighbors[nk2]:
					pass 
				else:
					neighbors[nk2].append(nk1)
			else:
				neighbors[nk2] = [nk1]

	g = graph_refine(neighbors, isolated_thr = 32, spurs_thr=0)
	g = simpilfyGraph(g, e=2)

	dim = np.shape(im)
	img = np.zeros((dim[0], dim[1]), dtype= np.uint8) + 255
	for nloc, nei in g.items():
		x1,y1 = int(nloc[1]), int(nloc[0])
		for nn in nei:
			x2,y2 = int(nn[1]), int(nn[0])
			cv2.line(img, (y1,x1), (y2,x2), (127), 2)
			cv2.circle(img, (y1,x1), 2, (0), -1)
			cv2.circle(img, (y2,x2), 2, (0), -1)
	cv2.imwrite(os.path.join(output_dir,'pred_graph_vis',name+'_with_node.png'), img)

	output_edges = []
	img = np.zeros((dim[0], dim[1]), dtype= np.uint8)
	for nloc, nei in g.items():
		x1,y1 = int(nloc[1]), int(nloc[0])
		for nn in nei:
			x2,y2 = int(nn[1]), int(nn[0])
			cv2.line(img, (y1,x1), (y2,x2), (255), 1)
			output_edges.append([[int(y1),int(x1)],[int(y2),int(x2)]])
	cv2.imwrite(os.path.join(output_dir,'pred_graph_vis',name+'.png'), img)

	if save_graph:
		with open(os.path.join(output_dir,'pred_graph',name+'.json'),'w') as jf:
			json.dump(output_edges,jf)

		graph = Graph()
		for e in output_edges:
			graph.add(e)

		output_graph = {}
		for k, v in graph.vertices.items():
			output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]

		pickle.dump(output_graph,open(os.path.join(output_dir,'pred_graph',name+'.p'),'wb'),protocol=2)
	
	return img















