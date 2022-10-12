import numpy as np 

def graph_refine(graph, isolated_thr = 50 * 3, spurs_thr = 20 * 3, three_edge_loop_thr = 70):
	neighbors = graph

	gid = 0 
	grouping = {}

	for k, v in neighbors.items():
		if k not in grouping:
			# start a search 

			queue = [k]

			while len(queue) > 0:
				n = queue.pop(0)

				if n not in grouping:
					grouping[n] = gid 
					for nei in neighbors[n]:
						queue.append(nei)

			gid += 1 

	group_count = {}

	for k, v in grouping.items():
		if v not in group_count:
			group_count[v] = (1,0)
		else:
			group_count[v] = (group_count[v][0] + 1, group_count[v][1])


		for nei in neighbors[k]:
			a = k[0] - nei[0]
			b = k[1] - nei[1]

			d = np.sqrt(a*a + b*b)

			group_count[v] = (group_count[v][0], group_count[v][1] + d/2)

	# short spurs
	remove_list = []
	for k, v in neighbors.items():
		if len(v) == 1:
			if len(neighbors[v[0]]) >= 3:
				a = k[0] - v[0][0]
				b = k[1] - v[0][1]

				d = np.sqrt(a*a + b*b)	

				if d < spurs_thr:
					remove_list.append(k)


	remove_list2 = []
	remove_counter = 0
	new_neighbors = {}

	def isRemoved(k):
		gid = grouping[k]
		if group_count[gid][0] <= 1:
			return True 
		elif group_count[gid][1] <= isolated_thr:
			return True 
		elif k in remove_list:
			return True 
		elif k in remove_list2:
			return True
		else:
			return False

	for k, v in neighbors.items():
		if isRemoved(k): 
			remove_counter += 1
			pass
		else:
			new_nei = []
			for nei in v:
				if isRemoved(nei):
					pass 
				else:
					new_nei.append(nei)

			new_neighbors[k] = list(new_nei)

	#print(len(new_neighbors), "remove", remove_counter, "nodes")

	return new_neighbors

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


def downsample(graph, rate = 2):
	newgraph = {}
	for nid, nei in graph.items():
		new_nid = ((int(nid[0])//rate) * rate,(int(nid[1])//rate) * rate)
		for nn in nei:
			new_nn = ((int(nn[0])//rate) * rate,(int(nn[1])//rate) * rate)	
			newgraph = graphInsert(newgraph, new_nid, new_nn)

	return newgraph	

def connectDeadEnds(graph, thr = 30):
	deadends = []
	for nloc, nei in graph.items():
		if len(nei) == 1:
			deadends.append(nloc)

	
	for d1 in deadends:
		cloest = None 
		bestd = thr

		for d2 in deadends:
			if d2 == d1 :
				continue 
			d  = np.sqrt((d1[0] - d2[0])**2 + (d1[1] - d2[1])**2 )
			if d < bestd:
				bestd = d 
				cloest = d2 
		
		if cloest is not None:
			graph = graphInsert(graph, d1, cloest)
	return graph