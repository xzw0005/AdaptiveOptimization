'''
Created on Oct 24, 2016

@author: XING
'''
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

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




if __name__ == '__main__':
    pass