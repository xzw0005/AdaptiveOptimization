'''
Created on Oct 26, 2016

@author: Xing Wang
'''
import numpy as np
import random
import copy
import numbers
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
# w1 = open('wd.txt', 'w')
# for line in distance_matrix:
#     w1.write(str(line)+',\n')
# w1.close()
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
Q = 1.

def costBetweenTwoDepartments(sol, i, j):
    ''' i, j are location indices '''
    #if (i < j):
    #    temp = i; i = j; j = temp;
    distance = distance_matrix[i, j]
    dept1 = sol[i]
    dept2 = sol[j]
    flow = flow_matrix[dept1, dept2]
    return flow * distance

def computeTotalCost(sol):
    #assert sol is np.ndarray
    totalCost = 0
    for loc1 in np.arange(1, N):
        for loc2 in np.arange(loc1):
            totalCost = totalCost + costBetweenTwoDepartments(sol, loc1, loc2)
    return totalCost

def trailHeuristic(distance_matrix, flow_matrix):
    distance_potentials_vector = np.sum(distance_matrix, axis=1)    # sum over row
    flow_potentials_vector = np.sum(flow_matrix, axis=1)            # sum over row
    coupling_matrix = np.outer(distance_potentials_vector, flow_potentials_vector)
    return 1.0/coupling_matrix

def initializeQAP(num_ants, seed=None):
    if seed is None:
        rng = np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer)):
        rng = np.random.RandomState(seed)
    else:
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)
    colony = []
    for i in range(num_ants):
        sol = rng.permutation(N)
        ant = Ant(sol)
        colony.append(ant)
    colony.sort(key = lambda ant: ant.fval)     # sort the ants by their cost from lowest to highest
    tau0 = 1e2 / (Q * colony[0].fval)
    pheromone_matrix = np.ones((N, N)) * tau0
    #print pheromone_matrix
    return colony, pheromone_matrix
    
def constructSolution(pheromone_matrix, heuristic_matrix, alpha, beta, q0):
    N = len(heuristic_matrix)
    unassigned = np.random.permutation(N)
    unassigned = [i for i in unassigned]
    #sol = np.zeros(N)
    sol=[]
    #order = np.random.permutation(N)
    #for i in order:
    for i in range(N):
        tot = 0.
        prob = []
        for j in unassigned:
            val = (pheromone_matrix[i][j] ** alpha) * (heuristic_matrix[i][j] ** beta)
            prob.append(val)
            tot += val
        q = np.random.uniform()
        if (q >= q0):
            prob = [val / tot for val in prob]
            cdf = np.cumsum(prob)
            u = np.random.uniform()
            idx = cdf.searchsorted(u)
        else:
            idx = np.argmax(prob)
        k = unassigned[idx]
        sol.append(k)
        unassigned.remove(k)
    #print sol
    return sol


class Ant(object):
    def __init__(self, sol):
        self.sol = sol
        self.fval = self.evaluate(sol)
        self.delta_pheromone =  self.get_delta_pheromone(self.sol, self.fval)
        
    def evaluate(self, sol):
        return computeTotalCost(sol)
    
    def get_delta_pheromone(self, sol, fval):
        N = len(sol)
        delta_pheromone = {}
        for i in range(N):
            j = sol[i]
            # (i, j) is the location-department pair
            delta_pheromone[(i, j)] = 1. / (Q * fval)
        return delta_pheromone

def renew_ant(pheromone_matrix, heuristic_matrix, alpha, beta, q0):
    sol = constructSolution(pheromone_matrix, heuristic_matrix, alpha, beta, q0)
    return Ant(sol)

def updatePheromoneByAnt(pheromone_matrix, ant, rho):
    for key in ant.delta_pheromone:
        i, j = key
        pheromone_matrix[i][j] += rho * ant.delta_pheromone[i, j]
    return pheromone_matrix

def aco(seed=123, max_iter=5e3, num_ants=20, rho=0.1, alpha=1., beta=0., q0=0.):
    heuristic_matrix = trailHeuristic(distance_matrix, flow_matrix)
    print heuristic_matrix
    colony, pheromone_matrix = initializeQAP(num_ants, seed)
    best_ant = copy.copy(colony[0])
    best_history = []
    iter = 0
    while (iter < max_iter and best_ant.fval > 1285):
        iter += 1
#         for ant in colony:
#             if (ant.fval <= best_ant.fval):
#                 best_ant = copy.copy(ant)
        if (colony[0].fval <= best_ant.fval):
            best_ant = copy.copy(colony[0])
            #ant.delta_pheromone = ant.get_delta_pheromone(ant.sol, ant.fval)
        pheromone_matrix = (1 - rho) * pheromone_matrix
        #for ant in colony:
        #    pheromone_matrix = updatePheromoneByAnt(pheromone_matrix, ant, rho)
        pheromone_matrix = updatePheromoneByAnt(pheromone_matrix, best_ant, rho)
#         print '############################ iter = %d ##############################'%iter
#         print pheromone_matrix
        colony = []
        for i in range(num_ants):
            ant = renew_ant(pheromone_matrix, heuristic_matrix, alpha, beta, q0)
            colony.append(ant)
        colony.sort(key = lambda ant: ant.fval)     # sort the ants by their cost from lowest to highest
        #print [(ant.sol, ant.fval) for ant in colony]
        print colony[0].sol, colony[0].fval
        best_history.append(best_ant.fval)
    return best_ant, best_history

if __name__ == '__main__':
    #colony, pheromone_matrix = initializeQAP(10, 0)
    #print [colony[i].fval for i in range(len(colony))]
    #print pheromone_matrix
    best_ant, best_history = aco(seed=123, max_iter=1e3, num_ants=100, rho=0.02, alpha=1., beta=0., q0=0.2)
    print best_history