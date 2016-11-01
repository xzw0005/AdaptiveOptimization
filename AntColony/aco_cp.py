import numpy as np
import copy

distMat = [
[0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7],
[1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6],
[2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5],
[3, 2, 1, 0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3, 6, 5, 4, 3, 4],
[4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3],
[1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
[2, 1, 2, 3, 4, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5],
[3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4],
[4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3],
[5, 4, 3, 2, 1, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2],
[2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
[3, 2, 3, 4, 5, 2, 1, 2, 3, 4, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4],
[4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3],
[5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 4, 3, 2, 1, 2],
[6, 5, 4, 3, 2, 5, 4, 3, 2, 1, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1],
[3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4],
[4, 3, 4, 5, 6, 3, 2, 3, 4, 5, 2, 1, 2, 3, 4, 1, 0, 1, 2, 3],
[5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2],
[6, 5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1],
[7, 6, 5, 4, 3, 6, 5, 4, 3, 2, 5, 4, 3, 2, 1, 4, 3, 2, 1, 0]]
distMat = np.array(distMat)

flowMat = [
[0, 0, 5, 0, 5, 2, 10, 3, 1, 5, 5, 5, 0, 0, 5, 4, 4, 0, 0, 1],
[0, 0, 3, 10, 5, 1, 5, 1, 2, 4, 2, 5, 0, 10, 10, 3, 0, 5, 10, 5],
[5, 3, 0, 2, 0, 5, 2, 4, 4, 5, 0, 0, 0, 5, 1, 0, 0, 5, 0, 0],
[0, 10, 2, 0, 1, 0, 5, 2, 1, 0, 10, 2, 2, 0, 2, 1, 5, 2, 5, 5],
[5, 5, 0, 1, 0, 5, 6, 5, 2, 5, 2, 0, 5, 1, 1, 1, 5, 2, 5, 1],
[2, 1, 5, 0, 5, 0, 5, 2, 1, 6, 0, 0, 10, 0, 2, 0, 1, 0, 1, 5],
[10, 5, 2, 5, 6, 5, 0, 0, 0, 0, 5, 10, 2, 2, 5, 1, 2, 1, 0, 10],
[3, 1, 4, 2, 5, 2, 0, 0, 1, 1, 10, 10, 2, 0, 10, 2, 5, 2, 2, 10],
[1, 2, 4, 1, 2, 1, 0, 1, 0, 2, 0, 3, 5, 5, 0, 5, 0, 0, 0, 2],
[5, 4, 5, 0, 5, 6, 0, 1, 2, 0, 5, 5, 0, 5, 1, 0, 0, 5, 5, 2],
[5, 2, 0, 10, 2, 0, 5, 10, 0, 5, 0, 5, 2, 5, 1, 10, 0, 2, 2, 5],
[5, 5, 0, 2, 0, 0, 10, 10, 3, 5, 5, 0, 2, 10, 5, 0, 1, 1, 2, 5],
[0, 0, 0, 2, 5, 10, 2, 2, 5, 0, 2, 2, 0, 2, 2, 1, 0, 0, 0, 5],
[0, 10, 5, 0, 1, 0, 2, 0, 5, 5, 5, 10, 2, 0, 5, 5, 1, 5, 5, 0],
[5, 10, 1, 2, 1, 2, 5, 10, 0, 1, 1, 5, 2, 5, 0, 3, 0, 5, 10, 10],
[4, 3, 0, 1, 1, 0, 1, 2, 5, 0, 10, 0, 1, 5, 3, 0, 0, 0, 2, 0],
[4, 0, 0, 5, 5, 1, 2, 5, 0, 0, 0, 1, 0, 1, 0, 0, 0, 5, 2, 0],
[0, 5, 5, 2, 2, 0, 1, 2, 0, 5, 2, 1, 0, 5, 5, 0, 5, 0, 1, 1],
[0, 10, 0, 5, 5, 1, 0, 2, 0, 5, 2, 2, 0, 5, 10, 2, 2, 1, 0, 6],
[1, 5, 0, 5, 1, 5, 10, 10, 2, 2, 5, 5, 5, 0, 10, 0, 0, 1, 6, 0]]
flowMat = np.array(flowMat)
N = len(flowMat)
Q = 1.e-3

class Ant(object):
    def __init__(self, sol):
        self.sol = sol
        self.fval = total_cost(sol)
        self.dtau =  self.get_dtau(self.sol, self.fval)
    
    def get_dtau(self, sol, fval):
        N = len(sol)
        dtau = {}
        for i in range(N):
            j = sol[i]
            dtau[(i, j)] = 1. / (1. * fval)
        return dtau

def total_cost(sol):
    total = 0
    for i in np.arange(1, N):
        for j in np.arange(i):
            dist = distMat[i, j]
            p = sol[i]
            q = sol[j]
            flow = flowMat[p,q]
            total = total + flow * dist 
    return total

def initial(num_ants, seed):
    rng = np.random.RandomState(seed)
    population = []
    for i in range(num_ants):
        sol = rng.permutation(N)
        ant = Ant(sol)
        population.append(ant)
    population.sort(key = lambda ant: ant.fval)
    tauMat = np.ones((N, N)) / (Q * population[0].fval)
    return population, tauMat
    
def new_perm(pheromone_matrix, heuristic_matrix, alpha, beta, q0):
    N = len(heuristic_matrix)
    unassigned = np.random.permutation(N)
    unassigned = [i for i in unassigned]
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

def new_ant(tauMat, etaMat, alpha, beta, q0):
    sol = new_perm(tauMat, etaMat, alpha, beta, q0)
    return Ant(sol)

def update_tau(tauMat, ant, rho):
    for key in ant.dtau:
        i, j = key
        tauMat[i][j] += ant.dtau[i, j] * rho
    return tauMat

def run(seed=0, max_iter=5e3, popsize=20, rho=0.1, alpha=1., beta=0., q0=0.):
    distsum = np.sum(distMat, axis=1)  
    flowsum = np.sum(flowMat, axis=1)  
    etaMat = 1.0/np.outer(distsum, flowsum)
    population, tauMat = initial(popsize, seed)
    best = copy.copy(population[0])
    best_history = []
    iter = 0
    while (iter < max_iter):
        iter += 1
        if (population[0].fval <= best.fval):
            best = copy.copy(population[0])
        tauMat = (1 - rho) * tauMat
        tauMat = update_tau(tauMat, best, rho)
        population = []
        for i in range(popsize):
            ant = new_ant(tauMat, etaMat, alpha, beta, q0)
            population.append(ant)
        population.sort(key = lambda ant: ant.fval) 
        print population[0].sol, population[0].fval
        best_history.append(best.fval)
    return best, best_history

if __name__ == '__main__':
    best, best_history = run(seed=0, max_iter=500, popsize=100, rho=0.02, alpha=1., beta=.8, q0=0.2)
    print best_history