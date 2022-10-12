import math 
import numpy as np 

def distance(p1,p2):
	a = p1[0]-p2[0]
	b = p1[1]-p2[1]

	return math.sqrt(a*a+b*b)


def point2lineDistance(p, n1, n2):
	l = distance(n1,n2)

	v1 = [n1[0]-p[0], n1[1]-p[1]]
	v2 = [n2[0]-p[0], n2[1]-p[1]]
	
	area = abs(v1[0]*v2[1]-v1[1]*v2[0])

	return area/l


def douglasPeucker(node_list, e = 5):
	new_list = []

	if len(node_list) <= 2:
		return node_list

	best_i = 1
	best_d = 0

	for i in range(1, len(node_list)-1):
		d = point2lineDistance(node_list[i], node_list[0], node_list[-1])
		if d > best_d:
			best_d = d 
			best_i = i 

	if best_d <= e:
		return [node_list[0], node_list[-1]]

	new_list = douglasPeucker(node_list[0:best_i+1], e=e)
	new_list = new_list[:-1] + douglasPeucker(node_list[best_i:len(node_list)], e=e)

	return new_list




def graphInsert(node_neighbor, n1key, n2key):
	if n1key != n2key:
		if n1key in node_neighbor:
			if n2key in node_neighbor[n1key]:
				pass 
			else:
				node_neighbor[n1key].append(n2key)
		else:
			node_neighbor[n1key] = [n2key]


		if n2key in node_neighbor:
			if n1key in node_neighbor[n2key]:
				pass 
			else:
				node_neighbor[n2key].append(n1key)
		else:
			node_neighbor[n2key] = [n1key]

	return node_neighbor



def simpilfyGraph(node_neighbor, e=2.5):
	new_graph = {}	

	visited = []

	new_node_neighbor = {}

	for node, node_nei in node_neighbor.items():
		if len(node_nei) == 1 or len(node_nei) > 2:
			if node in visited:
				continue

			# search node_nei 
			for next_node in node_nei:
				if next_node in visited:
					continue

				node_list = [node, next_node]

				current_node = next_node 

				while True:
					if len(node_neighbor[node_list[-1]]) == 2:
						if node_neighbor[node_list[-1]][0] == node_list[-2]:
							node_list.append(node_neighbor[node_list[-1]][1])
						else:
							node_list.append(node_neighbor[node_list[-1]][0])
					else:
						break

				for i in range(len(node_list)-1):
					if node_list[i] not in visited:
						visited.append(node_list[i])

				# simplify node_list
				new_node_list = douglasPeucker(node_list, e=e)

				for i in range(len(new_node_list)-1):
					new_node_neighbor = graphInsert(new_node_neighbor, new_node_list[i],new_node_list[i+1])

	return new_node_neighbor


def colorGraph(node_neighbor, colormap):
		
	visited = []

	edge_color = {}

	for node, node_nei in node_neighbor.items():
		if len(node_nei) == 1 or len(node_nei) > 2:
			if node in visited:
				continue

			# search node_nei 
			for next_node in node_nei:
				if next_node in visited:
					continue

				node_list = [node, next_node]

				current_node = next_node 

				while True:
					if len(node_neighbor[node_list[-1]]) == 2:
						if node_neighbor[node_list[-1]][0] == node_list[-2]:
							node_list.append(node_neighbor[node_list[-1]][1])
						else:
							node_list.append(node_neighbor[node_list[-1]][0])
					else:
						break

				for i in range(len(node_list)-1):
					if node_list[i] not in visited:
						visited.append(node_list[i])

				# node_list 
				counter = np.zeros((3))
				for i in range(len(node_list)-1):
					x1,y1 = node_list[i]
					x2,y2 = node_list[i+1]

					l = np.sqrt((x1-x2)**2 + (y1-y2)**2)
					n = max(2, int(l / 10.0))
					samples = []
					for j in range(n):
						a = float(j+1) / (n+1)
					
						x3 = int(x1 * a + x2 * (1-a))
						y3 = int(y1 * a + y2 * (1-a))
						samples.append((x3,y3))

					if i != 0:
						samples.append((x1,y1))

					for sample in samples:
						x,y = sample

						if x > 5 and x < np.shape(colormap)[0]-5 and y > 5 and y < np.shape(colormap)[1]-5:
							counter[0] += np.sum(colormap[x-3:x+4,y-3:y+4,0])
							counter[1] += np.sum(colormap[x-3:x+4,y-3:y+4,1])
							counter[2] += np.sum(colormap[x-3:x+4,y-3:y+4,2])
				
				color = np.argmax(counter)
				s = counter[0] + counter[1] + counter[2] + 0.0001
				counter = [float(v) / s for v in counter]

				for i in range(len(node_list)-1):
					x1,y1 = node_list[i]
					x2,y2 = node_list[i+1]

					edgekey1 = ((x1,y1), (x2,y2))
					edgekey2 = ((x2,y2), (x1,y1))

					edge_color[edgekey1] = list(counter) 
					edge_color[edgekey2] = list(counter)
					
	return edge_color


				