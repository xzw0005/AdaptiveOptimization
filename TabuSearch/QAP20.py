'''
Created on Oct 3, 2016
@author: Xing Wang
'''
import numpy as np

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
#print flow_matrix.shape

def costBetweenTwoDepartments(X, i, j):
    ''' i, j are location indices '''
    #if (i < j):
    #    temp = i; i = j; j = temp;
    distance = distance_matrix[i, j]
    dept1 = X[i]
    dept2 = X[j]
    flow = flow_matrix[dept1, dept2]
    return flow * distance

def computeTotalCost(X):
    #assert X is np.ndarray
    totalCost = 0
    for loc1 in np.arange(1, len(flow_matrix)):
        for loc2 in np.arange(loc1):
            totalCost = totalCost + costBetweenTwoDepartments(X, loc1, loc2)
    return totalCost

print 'The optimal solution is %d' %(2570/2)
np.random.seed(123)
ten_seeds = np.random.randint(0, 1e3, size=10)
#print ten_seeds
Xs = []
for i in range(len(ten_seeds)):
    np.random.seed(ten_seeds[i])
    X = np.random.permutation(len(flow_matrix))
    print computeTotalCost(X)
    Xs.append(X)
#print Xs
