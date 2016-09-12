'''
Created on Sep 12, 2016

@author: XING
'''
import GeneticAlgorithms.prod.Individual as Individual
import numpy as np

class TenInitialPopulations(object):
    '''
    Generate 10 initial populations for GA experiments
    '''

    def __init__(self, ps = None):
        '''
        Constructor
        '''
        np.random.seed(123456)
        tenSeeds = np.random.randint(0, 100, size=10)
        
        init10 = []
     
        populationSize = 2000
        for sd in tenSeeds:
            np.random.randint(sd)
            indivSeeds = np.random.randint(0, 1e6, size = populationSize)
            population = []
            for seed in indivSeeds:
                member = Individual.Individual()
                member.randomIndividual(seed)
                population.append(member)   
            init10.append(population)
        
        
        sln576a = np.array([11,  8,  7, 12, 10,  9, 13,  3,  5, 15,  1,  2,  4, 14,  6])-1
        sln575 = np.array([1,  2, 13,  8,  9,  4,  3, 14,  7, 11, 10, 15,  6,  5, 12])-1
        sln576b = np.array([ 6, 14,  4,  2,  1, 15,  5,  3, 13,  9, 10, 12,  7,  8, 11])-1
        sln576c = np.array([1,  2,  4, 14,  6,  9, 13,  3,  5, 15, 11,  8,  7, 12, 10])-1
        sln580a = np.array([15,  5, 12,  9, 11,  6, 14,  7, 13,  8, 10,  3,  4,  2,  1])-1 #stuck easily
        sln580b = np.array([1,  2,  4,  3, 10,  8, 13,  7, 14,  6, 11,  9, 12,  5, 15])-1 #stuck

        init10[0][0].chromosome = sln575
        
        init10[1][0].chromosome = sln575
        init10[1][1].chromosome = sln580a
        
        init10[2][0].chromosome = sln576a
        init10[2][1].chromosome = sln576b
        init10[2][2].chromosome = sln576c
        init10[2][3].chromosome = sln580a
        init10[2][4].chromosome = sln580b
        
        init10[3][0].chromosome = sln580a
        
        init10[4][0].chromosome = sln576b
        init10[4][1].chromosome = sln580b
        
        if ps is not None:
            for i in np.arange(len(init10)):
                init10[i] = init10[i][0:ps]
                
        self.init10 = init10
        

#         population = []
#         while populationSize:
#             member = Individual.Individual()
#             #member.randomIndividual()
#             population.append(member)
#             populationSize -= 1
#         self.population = population
        
    