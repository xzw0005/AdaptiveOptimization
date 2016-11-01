'''
Created on Oct 31, 2016

@author: XING
'''

import numpy as np
import random
import copy
import numbers
import time
import matplotlib
import matplotlib.pyplot as plt

fd = open('distMat.txt', 'r')
line = fd.readline()
distMat = []
while (line):
    ll = line.rstrip().split()
    dists_row = [int(d) for d in ll]
    distMat.append(dists_row)
    line = fd.readline()
fd.close()
distMat = np.array(distMat)
#print distMat.shape

ff = open('flowMat.txt', 'r')
line = ff.readline()
flowMat = []
while (line):
    ll = line.rstrip().split()
    flows_row = [int(flow) for flow in ll]
    flowMat.append(flows_row)
    line = ff.readline()
ff.close()
flowMat = np.array(flowMat)
N = len(flowMat)
#print flowMat.shape
Q = 1e3

def between_cost(sol, i, j):
    ''' i, j are location indices '''
    #if (i < j):
    #    temp = i; i = j; j = temp;
    distance = distMat[i, j]
    dept1 = sol[i]
    dept2 = sol[j]
    flow = flowMat[dept1, dept2]
    return flow * distance

def total_cost(sol):
    #assert sol is np.ndarray
    totalCost = 0
    for loc1 in np.arange(1, N):
        for loc2 in np.arange(loc1):
            totalCost = totalCost + between_cost(sol, loc1, loc2)
    return totalCost

def getNeighbor(sol, i, j):
    x= copy.copy(sol)
    temp = x[i]
    x[i] = x[j]
    x[j] = temp
    return x      

def heuristic(distMat, flowMat):
    distance_potentials_vector = np.sum(distMat, axis=1)    # sum over row
    flow_potentials_vector = np.sum(flowMat, axis=1)            # sum over row
    coupling_matrix = np.outer(distance_potentials_vector, flow_potentials_vector)
    return 1.0/coupling_matrix

def initial(num_ants, seed=None):
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
    tau0 = 1 / (Q * colony[0].fval)
    pheromone_matrix = np.ones((N, N)) * tau0
    #print pheromone_matrix
    return colony, pheromone_matrix
 
def new_ant(tauMat, etaMat, alpha, beta, old_ant, q0):
    N = len(etaMat)
    old_sol = old_ant.sol
    r = np.random.choice(range(N))
    tot = 0.
    prob = []
    for i in range(N):
        if (i != r):
            tau = tauMat[r][old_sol[i]] + tauMat[i][old_sol[r]]
            eta = etaMat[r][old_sol[i]] + etaMat[i][old_sol[r]]
            val = (tau ** alpha) * (eta ** beta)
            prob.append(val)
            tot += val
    #print prob
    q = np.random.uniform()
    if (q >= q0):
        prob = [val / tot for val in prob]
        cdf = np.cumsum(prob)
        u = np.random.uniform()
        s = cdf.searchsorted(u)
    else:
        s = np.argmax(prob)
    #sol = getNeighbor(old_sol, r, s)
    #return Ant(sol)
    return r, s

class Ant(object):
    def __init__(self, sol):
        self.sol = sol
        self.fval = self.evaluate(sol)
#         self.delta_pheromone =  self.get_delta_pheromone(self.sol, self.fval)
        
    def evaluate(self, sol):
        return total_cost(sol)
    
# def get_delta_pheromone(r, s, fval):
#     delta_pheromone = {}
#         # (i, j) is the location-department pair
#         delta_pheromone[(i, j)] = 1. / (Q * fval)
#     return delta_pheromone

# def updatePheromoneByAnt(pheromone_matrix, ant, rho):
#     for key in ant.delta_pheromone:
#         i, j = key
#         pheromone_matrix[i][j] += rho * ant.delta_pheromone[i, j]
#     return pheromone_matrix

def run(seed=123, max_iter=5e3, popsize=20, rho=0.1, alpha=1., beta=0., q0=0.):
    heuristic_matrix = heuristic(distMat, flowMat)
    colony, pheromone_matrix = initial(popsize, seed)
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
#         for ant in colony:
#             pheromone_matrix = updatePheromoneByAnt(pheromone_matrix, ant, rho)
#         pheromone_matrix = updatePheromoneByAnt(pheromone_matrix, best_ant, rho)
#         print '############################ iter = %d ##############################'%iter
#         print pheromone_matrix
        new_colony = []
        for i in range(popsize):
            old_ant = colony[i]
            r, s = new_ant(pheromone_matrix, heuristic_matrix, alpha, beta, old_ant, q0)
            new_sol = getNeighbor(old_ant.sol, r, s)
            fval = total_cost(new_sol)
            pheromone_matrix[r][s] += 1. / (Q * fval)
            new_colony.append(Ant(new_sol))
        colony = copy.copy(new_colony)
        colony.sort(key = lambda ant: ant.fval)     # sort the ants by their cost from lowest to highest
        #print [(ant.sol, ant.fval) for ant in colony]
        print colony[0].sol, colony[0].fval
        best_history.append(best_ant.fval)
    return best_ant, best_history

if __name__ == '__main__':
    #colony, pheromone_matrix = initial(10, 0)
    #print [colony[i].fval for i in range(len(colony))]
    #print pheromone_matrix
    best_ant, best_history = run(seed=123, max_iter=5e3, popsize=50, rho=0.9, alpha=1., beta=0., q0=0.0)
    print best_history