'''
Created on Sep 9, 2016
@author: XING
'''
import numpy as np

class Individual(object):       
    N = 15
    FLOW_MATRIX = np.zeros((N, N), dtype = np.int)
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

    def __init__(self, chromosome = None):
        '''
        Constructor
        '''
        if (chromosome is not None):
            self.chromosome = chromosome
            self.cost = self.total_cost(chromosome)
            self.fitness = self.getFitness(self.cost)
        else:
            self.chromosome = None
            self.cost = None
            self.fitness = None
        
    def randomIndividual(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
        self.chromosome = np.random.permutation(Individual.N)
        self.cost = self.total_cost(self.chromosome)
        self.fitness = self.getFitness(self.cost)

    def computeDistance(self, i, j):
        ''' i, j are location indices '''
        return abs(j % 5 - i % 5) + abs(j / 5 - i / 5)

    def between_cost(self, X, i, j):
        ''' i, j are location indices '''
        distance = self.computeDistance(i, j)
        dept1 = X[i]
        dept2 = X[j]
        if (dept1 < dept2):
            temp = dept1; dept1 = dept2; dept2 = temp;
        flow = Individual.FLOW_MATRIX[dept1, dept2]
        return flow * distance
    
    def total_cost(self, X):
        #assert X is np.ndarray
        totalCost = 0
        for loc1 in np.arange(1, Individual.N):
            for loc2 in np.arange(loc1):
                totalCost = totalCost + self.between_cost(X, loc1, loc2)
        return totalCost
        
    def getFitness(self, cost):
        return 1.0/cost


    def uniformCrossover(self, parent1, parent2, randominzed = False):
        ''' Both parent1 & parent2 are numpy arrays'''
        child = np.zeros(Individual.N, dtype = np.int) - 1
        if (randominzed == True):
            order = np.random.permutation(Individual.N)
        else:
            order = np.arange(Individual.N)
        emptyPosition = []
        for i in order:
            if (parent1[i] not in child) and (parent2[i] not in child):
                u = np.random.uniform()
                if (u >= 0.5):
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            elif (parent1[i] not in child) and (parent2[i] in child):
                child[i] = parent1[i]
            elif (parent1[i] in child) and (parent2[i] not in child):
                child[i] = parent2[i]
            else:
                emptyPosition.append(i)
        if (len(emptyPosition) != 0):
            leftover = []
            for i in np.arange(Individual.N):
                if (i not in child):
                    leftover.append(i)
            #print len(emptyPosition), len(leftover)
            assert len(emptyPosition) == len(leftover)
            repairOrder = np.random.permutation(len(leftover))
            for i in np.arange(len(emptyPosition)):
                repairPosition = emptyPosition[i]
                j = repairOrder[i]
                child[repairPosition] = leftover[j]
        return child
                
    def singlePointCrossover(self, parent1, parent2):
        crossoverPoint = np.random.randint(1, Individual.N)
        return None
        
    def swapMutation(self, chromosome):
        x= np.copy(chromosome)
        locPairs = np.random.choice(Individual.N, 2, replace=False)
        i = locPairs[0]; j = locPairs[1];
        temp = x[i]
        x[i] = x[j]
        x[j] = temp
        return x      
    
    def shiftMutation(self, chromosome):
        Y = np.copy(chromosome)
        locPairs = np.random.choice(Individual.N, 2, replace=False).sort()
        i = locPairs[0]; j = locPairs[1];
        temp = Y[j]
        for k in np.arange(j, i, -1):
            Y[k] = Y[k - 1]
        Y[i] = temp
        return Y     

    def __repr__(self):
        return self.chromosome.__repr__()

#    def __str__(self):
#         string = "["
#         for i in self.chromosome:
#             string += str(i)
#             string += ", "
#         string += "]"
#         return string

#print Individual().randomIndividual()

# p1 = np.random.permutation(15)
# p2 = np.random.permutation(15)
# print p1,"\n", p2
# print ind.uniformCrossover(p1, p2, randominzed=True)
# print ind.swapMutation(p1)