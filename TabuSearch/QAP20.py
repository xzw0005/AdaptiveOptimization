'''
Created on Oct 3, 2016
@author: Xing Wang
'''
import numpy as np
import time

fd = open('distance_matrix.txt', 'r')
line = fd.readline()
distance_matrix = []
while (line):
    ll = line.rstrip().split()
    dists_row = [int(d) for d in ll]
    distance_matrix.append(dists_row)
    line = fd.readline()
fd.close()
distance_matrix = np.array(distance_matrix)
#print distance_matrix.shape

ff = open('flow_matrix.txt', 'r')
line = ff.readline()
flow_matrix = []
while (line):
    ll = line.rstrip().split()
    flows_row = [int(flow) for flow in ll]
    flow_matrix.append(flows_row)
    line = ff.readline()
ff.close()
flow_matrix = np.array(flow_matrix)
N = len(flow_matrix)
#print flow_matrix.shape

def costBetweenTwoDepartments(x, i, j):
    ''' i, j are location indices '''
    #if (i < j):
    #    temp = i; i = j; j = temp;
    distance = distance_matrix[i, j]
    dept1 = x[i]
    dept2 = x[j]
    flow = flow_matrix[dept1, dept2]
    return flow * distance

def computeTotalCost(x):
    #assert x is np.ndarray
    totalCost = 0
    for loc1 in np.arange(1, N):
        for loc2 in np.arange(loc1):
            totalCost = totalCost + costBetweenTwoDepartments(x, loc1, loc2)
    return totalCost

def compute_delta(flow_matrix, distance_matrix, x, i, j):
    d = (flow_matrix[i][i] - flow_matrix[j][j]) * (distance_matrix[x[j]][x[j]] - distance_matrix[x[i]][x[i]]) \
            + (flow_matrix[i][j] - flow_matrix[j][i] ) * (distance_matrix[x[j]][x[i]] - distance_matrix[x[i]][x[j]])
    for k in range(N):
        if (k != i and k != j):
            d += (flow_matrix[k][i] - flow_matrix[k][j]) * (distance_matrix[x[k]][x[j]] - distance_matrix[x[k]][x[i]]) \
                    + (flow_matrix[i][k] - flow_matrix[j][k] ) * (distance_matrix[x[j]][x[k]] - distance_matrix[x[i]][x[k]])
    return d

def getNeighbors(x):
    neighbors = []
    for i in range(N-1):
        for j in range(i+1, N):
            nb = np.copy(x)
            temp = nb[i]
            nb[i] = nb[j]
            nb[j] = temp
            neighbors.append([ [i, j], nb, computeTotalCost(nb)])
    neighbors.sort(key = lambda nb: nb[2])
    return neighbors

def tabuSearch(current_sol, maxlen = 8, maxiter=1e3):
    best_sol = np.copy(current_sol)
    best_val = computeTotalCost(best_sol)
    tabuList = []
    iter = 0
    while (best_val > 1285) and (iter < maxiter):
        neighbors = getNeighbors(current_sol)
        #nb0 = neighbors[0]
        for k in range(len(neighbors)):
            nblist = neighbors[k]
            (i, j) = nblist[0]
            loc_pair = sorted([current_sol[i], current_sol[j]])
            if (loc_pair not in tabuList):
                tabuList.append(loc_pair)
                if (len(tabuList) > maxlen):
                    tabuList.pop(0)
                current_sol = nblist[1]
                current_val = nblist[2]
                if (nblist[2] < best_val):
                    best_sol = nblist[1]
                    best_val = nblist[2]
                break
            if (k == len(neighbors)):
                raise ValueError
        iter += 1
        #print tabuList
        #print [iter, current_sol, current_val, best_sol, best_val]
    return [iter, best_sol, best_val]



print 'The optimal solution is %d' %(2570/2)
np.random.seed(123456)
ten_seeds = np.random.randint(0, 1e3, size=10)
#print ten_seeds
xs = []
for i in range(len(ten_seeds)):
    np.random.seed(ten_seeds[i])
    x = np.random.permutation(len(flow_matrix))
    #print computeTotalCost(x)
    print tabuSearch(x, maxlen=8, maxiter=1e3)
    xs.append(x)
#print xs
#print tabuSearch(xs[1], maxlen=8, maxiter=1e3)