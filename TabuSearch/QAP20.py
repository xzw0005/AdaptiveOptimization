'''
Created on Oct 3, 2016
@author: Xing Wang
'''
import numpy as np
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

def between_cost(x, i, j):
    ''' i, j are location indices '''
    #if (i < j):
    #    temp = i; i = j; j = temp;
    distance = distMat[i, j]
    dept1 = x[i]
    dept2 = x[j]
    flow = flowMat[dept1, dept2]
    return flow * distance

def total_cost(x):
    #assert x is np.ndarray
    totalCost = 0
    for loc1 in np.arange(1, N):
        for loc2 in np.arange(loc1):
            totalCost = totalCost + between_cost(x, loc1, loc2)
    return totalCost

def compute_delta(flowMat, distMat, x, i, j):
    d = (flowMat[i][i] - flowMat[j][j]) * (distMat[x[j]][x[j]] - distMat[x[i]][x[i]]) \
            + (flowMat[i][j] - flowMat[j][i] ) * (distMat[x[j]][x[i]] - distMat[x[i]][x[j]])
    for k in range(N):
        if (k != i and k != j):
            d += (flowMat[k][i] - flowMat[k][j]) * (distMat[x[k]][x[j]] - distMat[x[k]][x[i]]) \
                    + (flowMat[i][k] - flowMat[j][k] ) * (distMat[x[j]][x[k]] - distMat[x[i]][x[k]])
    return d

def getNeighbors(x, subset):
    neighbors = []
    for i in range(N-1):
        for j in range(i+1, N):
            nb = np.copy(x)
            temp = nb[i]
            nb[i] = nb[j]
            nb[j] = temp
            neighbors.append([ [i, j], nb, total_cost(nb)])
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
    best_val = total_cost(best_sol)
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
        x = np.random.permutation(len(flowMat))
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
    
#     colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
#     historyList = [res[-1] for res in results]
#     m = max([len(h) for h in historyList])
# #     for h in historyList:
# #         if len(h) < m:
# #             h = np.append(h, np.ones(m - len(h))*1285)
#     for i in range(len(historyList)):
#         plt.plot(np.arange(len(historyList[i])), historyList[i], c=colors[i])
#     plt.ylim([1285, 1350])
#     plt.show()  
    
    
if __name__ == '__main__':
    main()