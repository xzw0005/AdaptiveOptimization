'''
Created on Aug 29, 2016

@author: XING WANG    
'''

import numpy as np
import time

class QAP(object):
    '''
    Soving a Quadratic Assignment Problem of Department Locations, 
    by minimizing the total transportation through all departments.
    This is Q1 for HW1.
    '''
    FLOW_MATRIX = np.zeros((15, 15), dtype = np.int)
    FLOW_MATRIX[1, 0] = 10;
    FLOW_MATRIX[2, 1] = 1;
    FLOW_MATRIX[3, 0] = 5; FLOW_MATRIX[3, 1] = 3; FLOW_MATRIX[3, 2] = 10;
    FLOW_MATRIX[4, 0] = 1; FLOW_MATRIX[4, 1] = 2; FLOW_MATRIX[4, 2] = 2; FLOW_MATRIX[4, 3] = 1; 
    FLOW_MATRIX[5, 1] = 2; FLOW_MATRIX[5, 3] = 1; FLOW_MATRIX[5, 4] = 3; 
    FLOW_MATRIX[6, 0] = 1; FLOW_MATRIX[6, 1] = 2; FLOW_MATRIX[6, 2] = 2; FLOW_MATRIX[6, 3] = 5; FLOW_MATRIX[6, 4] = 5; FLOW_MATRIX[6, 5] = 2; 
    FLOW_MATRIX[7, 0] = 2; FLOW_MATRIX[7, 1] = 3; FLOW_MATRIX[7, 2] = 5; FLOW_MATRIX[7, 4] = 5; FLOW_MATRIX[7, 5] = 2; FLOW_MATRIX[7, 6] = 6;
    FLOW_MATRIX[8, 0] = 2; FLOW_MATRIX[8, 1] = 2; FLOW_MATRIX[8, 2] = 4; FLOW_MATRIX[8, 4] = 5; FLOW_MATRIX[8, 5] = 1; FLOW_MATRIX[8, 7] = 5;
    FLOW_MATRIX[9, 0] = 2; FLOW_MATRIX[9, 2] = 5; FLOW_MATRIX[9, 3] = 2; FLOW_MATRIX[9, 4] = 1; FLOW_MATRIX[9, 5] = 5; FLOW_MATRIX[9, 6] = 1; FLOW_MATRIX[9, 7] = 2;
    FLOW_MATRIX[10, 0] = 2; FLOW_MATRIX[10, 1] = 2; FLOW_MATRIX[10, 2] = 2; FLOW_MATRIX[10, 3] = 1; FLOW_MATRIX[10, 6] = 5; FLOW_MATRIX[10, 7] = 10; FLOW_MATRIX[10, 8] = 10; 
    FLOW_MATRIX[11, 2] = 2; FLOW_MATRIX[11, 4] = 3; FLOW_MATRIX[11, 6] = 5; FLOW_MATRIX[11, 8] = 5; FLOW_MATRIX[11, 9] = 4; FLOW_MATRIX[11, 10] = 5; 
    FLOW_MATRIX[12, 0] = 4; FLOW_MATRIX[12, 1] = 10; FLOW_MATRIX[12, 2] = 5; FLOW_MATRIX[12, 3] = 2; FLOW_MATRIX[12, 5] = 2; FLOW_MATRIX[12, 6] = 5; FLOW_MATRIX[12, 7] = 5; FLOW_MATRIX[12, 8] = 10; FLOW_MATRIX[12, 11] = 3; 
    FLOW_MATRIX[13, 1] = 5; FLOW_MATRIX[13, 2] = 5; FLOW_MATRIX[13, 3] = 5; FLOW_MATRIX[13, 4] = 5; FLOW_MATRIX[13, 5] = 5; FLOW_MATRIX[13, 6] = 1; FLOW_MATRIX[13, 10] = 5; FLOW_MATRIX[13, 11] = 3; FLOW_MATRIX[13, 12] = 10; 
    FLOW_MATRIX[14, 2] = 5; FLOW_MATRIX[14, 4] = 5; FLOW_MATRIX[14, 5] = 10; FLOW_MATRIX[14, 8] = 2; FLOW_MATRIX[14, 9] = 5; FLOW_MATRIX[14, 12] = 2; FLOW_MATRIX[14, 13] = 4; 


    def __init__(self, T0, m, n, alpha = 0.9, seed = 123456789):
        '''
        Constructor
        '''
        np.random.seed(seed)
        self.X0 = np.random.permutation(len(QAP.FLOW_MATRIX))
        self.T0 = T0
        self.m = m
        self.n = n
        self.alpha = alpha

    def computeDistance(self, i, j):
        ''' i, j are location indices '''
        return abs(j % 5 - i % 5) + abs(j / 5 - i / 5)

    def costBetweenTwoDepartments(self, X, i, j):
        ''' i, j are location indices '''
        #if (i < j):
        #    temp = i; i = j; j = temp;
        distance = self.computeDistance(i, j)
        dept1 = X[i]
        dept2 = X[j]
        if (dept1 < dept2):
            temp = dept1; dept1 = dept2; dept2 = temp;
        flow = QAP.FLOW_MATRIX[dept1, dept2]
        return flow * distance
    
    def computeTotalCost(self, X):
        #assert X is np.ndarray
        totalCost = 0
        for loc1 in np.arange(1, len(QAP.FLOW_MATRIX)):
            for loc2 in np.arange(loc1):
                totalCost = totalCost + self.costBetweenTwoDepartments(X, loc1, loc2)
        return totalCost
        
    def moveOperator(self, X):
        Y= np.copy(X)
        n = np.random.poisson(1, 1)[0] + 1
        for k in np.arange(n):
            locPairs = np.random.choice(len(QAP.FLOW_MATRIX), 2, replace=False)
            i = locPairs[0]; j = locPairs[1];
            temp = Y[i]
            Y[i] = Y[j]
            Y[j] = temp
        return Y
    
#     def moveOperator(self, X):
#         Y = np.copy(X)
#         return np.random.permutation(Y)
    
    def simulatedAnnealing(self):
        xOld = self.X0
        costOld = self.computeTotalCost(xOld)
        print "Initial Solution: ", xOld+1
        print "Initial Total Cost: ", costOld
        temperature = self.T0
        timesWithoutImprovement = 0
        for t in np.arange(self.m - 1):
            for i in range(self.n - 1):
                xNew = self.moveOperator(xOld)
                #print xNew
                costNew = self.computeTotalCost(xNew)
                deltaCost = costNew - costOld
                if (deltaCost <= 0):
                    timesWithoutImprovement = 0
                    xOld = np.copy(xNew)
                    costOld = costNew
                else:
                    timesWithoutImprovement = timesWithoutImprovement + 1
                    probabilityThreshold = np.exp(-deltaCost / temperature)
                    if (np.random.uniform() <= probabilityThreshold):
                        xOld = np.copy(xNew)
                        costOld = costNew
                #print xOld
                #if (timesWithoutImprovement >= 1e5):
                #    return xOld
                #print "Total Cost: ", costOld
            temperature = self.alpha * temperature
            if (temperature <= 1e-10):
                break
        return xOld

    
solveQAP = QAP(T0 = 100, m = 500, n = 500, alpha = 0.9, seed = 1)
print "T0 =", solveQAP.T0, ", m =", solveQAP.m, ", n =", solveQAP.n, ", alpha =", solveQAP.alpha, ", seed ="
startTime = time.clock()
res = solveQAP.simulatedAnnealing()  
print "Best Result Found: ", res+1
print "Minimum Total Cost Found: ", solveQAP.computeTotalCost(res)
endTime = time.clock()
print "Elapsed Time: ", endTime - startTime