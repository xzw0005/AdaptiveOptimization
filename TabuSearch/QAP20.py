'''
Created on Oct 3, 2016
@author: Xing Wang
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

def compute_delta(flow_matrix, distance_matrix, x, i, j):
    d = (flow_matrix[i][i] - flow_matrix[j][j]) * (distance_matrix[x[j]][x[j]] - distance_matrix[x[i]][x[i]]) \
            + (flow_matrix[i][j] - flow_matrix[j][i] ) * (distance_matrix[x[j]][x[i]] - distance_matrix[x[i]][x[j]])
    for k in range(N):
        if (k != i and k != j):
            d += (flow_matrix[k][i] - flow_matrix[k][j]) * (distance_matrix[x[k]][x[j]] - distance_matrix[x[k]][x[i]]) \
                    + (flow_matrix[i][k] - flow_matrix[j][k] ) * (distance_matrix[x[j]][x[k]] - distance_matrix[x[i]][x[k]])
    return d

def getNeighbors(x, subset):
    neighbors = []
    for i in range(N-1):
        for j in range(i+1, N):
            nb = np.copy(x)
            temp = nb[i]
            nb[i] = nb[j]
            nb[j] = temp
            neighbors.append([ [i, j], nb, computeTotalCost(nb)])
    if (subset < 1.):
        total_nbs = len(neighbors)  # N*(N-1)/2 
        num_nbs = int(total_nbs * subset)
        idxs = range(total_nbs)
        np.random.shuffle(idxs)
        idxs = idxs[:num_nbs]
        all_nbs = np.copy(neighbors)
        neighbors = [all_nbs[i] for i in idxs]
    neighbors.sort(key = lambda nb: nb[2])
    return neighbors

def dynamicTabuSize(iter, maxlen, tb_lo, tb_hi):
    if (iter % 50 == 0):
        maxlen = np.random.randint(tb_lo, tb_hi+1)
    return maxlen
    
def tabuSearch(current_sol, tabu_len = 8, maxiter=1e3, aspiration=False, subset=1.0, restart=False):
    best_sol = np.copy(current_sol)
    best_val = computeTotalCost(best_sol)
    best_history = [best_val]
    tabuList = []
    iter = 0
    if (type(tabu_len) is int):
        dynamic = False
        maxlen = tabu_len
    elif (len(tabu_len) == 2):
        dynamic = True
        tb_lo = min(tabu_len)
        tb_hi = max(tabu_len)
        maxlen = int((tb_lo+tb_hi)/2)
    else:
        raise ValueError('Invalid Tabu List Length, use an integer, or a 2-elements list or tuple')
    not_improved = 0
    while (best_val > 1285) and (iter < maxiter):
        neighbors = getNeighbors(current_sol, subset)
        if dynamic:
            maxlen = dynamicTabuSize(iter, maxlen, tb_lo, tb_hi)
        for k in range(len(neighbors)):
            neighbor = neighbors[k]
            (i, j) = neighbor[0]
            dept_pair = sorted([current_sol[i], current_sol[j]])
            this_sol = neighbor[1]
            this_val = neighbor[2]
            aspired = (aspiration and this_val < best_val)
            if (dept_pair not in tabuList) or (aspired):
                tabuList.append(dept_pair)
                if (len(tabuList) > maxlen):
                    tabuList = tabuList[-maxlen:]
                current_sol = this_sol
                current_val = this_val
                if (current_val < best_val):
                    best_sol = current_sol
                    best_val = current_val
                    not_improved = 0
                else:
                    not_improved += 1
                break
            if (k == len(neighbors)):
                raise ValueError
        if (restart == True) and (not_improved >= 300):
            current_sol = np.random.permutation(N)
            not_improved = 0
            tabuList = []
        iter += 1
        best_history.append(best_val)
        #print tabuList
        #print [iter, current_sol, current_val, best_sol, best_val]
    return [iter, best_sol, best_val, best_history]

def main():
    #print 'The optimal solution is %d' %(2570/2)
    np.random.seed(123456)
    ten_seeds = np.random.randint(0, 1e3, size=3)
    xs = []
    for i in range(len(ten_seeds)):
        np.random.seed(ten_seeds[i])
        x = np.random.permutation(len(flow_matrix))
        xs.append(x)
    xs = np.array(xs)
    results = []
    elapsedTimeList = []
    for x in xs:
        startTime = time.clock()
        res = tabuSearch(x, tabu_len=[6, 12], maxiter=.5e3, aspiration=True, subset=1., restart=False)
        elapsedTime = time.clock() - startTime
        elapsedTimeList.append(elapsedTime)
        results.append(res)
        print res[:-1]   
    for elapsedTime in elapsedTimeList:
        print 'Elapsed Time: %.3f' % elapsedTime
    print [res[:-1] for res in results]
    #print tabuSearch(xs[1], tabu_len=8, maxiter=1e3, aspiration=True, subset=1., restart=False)
    
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    historyList = [res[-1] for res in results]
    m = max([len(h) for h in historyList])
#     for h in historyList:
#         if len(h) < m:
#             h = np.append(h, np.ones(m - len(h))*1285)
    for i in range(len(historyList)):
        plt.plot(np.arange(len(historyList[i])), historyList[i], c=colors[i])
    plt.ylim([1285, 1350])
    plt.show()  
    
    
if __name__ == '__main__':
    main()