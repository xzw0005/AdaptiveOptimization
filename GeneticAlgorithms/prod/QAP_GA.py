'''
Created on Sep 5, 2016
@author: Xing Wang
'''

import numpy as np
import time

import GeneticAlgorithms.prod.Individual as Individual
#import GeneticAlgorithms.prod.Population as Population
import GeneticAlgorithms.prod.TenInitialPopulations as Init10
import GeneticAlgorithms.prod.TenInitialPopulations as TenInitialPopulations


class QAP_GA(object):
    '''
    Soving a Quadratic Assignment Problem of Department Locations, 
    by minimizing the total transportation through all departments.
    This is Q1 for HW2.
    '''

    def __init__(self, probCrossover, probMutation, maxGeneration, populationSize, initPopulation = None):
        '''
        Constructor
        '''
        if (initPopulation is None):
            self.populationSize = populationSize
            population = []
            while populationSize:
                member = Individual.Individual()
                member.randomIndividual()
                population.append(member)
                populationSize -= 1
            self.population = population            
        else:
            self.populationSize = len(initPopulation)
            self.population = initPopulation
        self.probCrossover = probCrossover
        self.probMutation = probMutation
        self.maxGeneration = maxGeneration
        self.generation = 0

    def sortPopulation(self, population):
        population.sort(key = lambda x: x.cost)
        return population
#         ranks = sorted(range(costList), key = lambda x: costList[x])
#         costList = [costList[i] for i in ranks]
#         fitList = [fitList[i] for i in ranks]
#         population = [population[i] for i in ranks]
#         return population, costList, fitList
    
    def rouletteWheelSelection(self, population):
        '''return 1 parent for crossover'''
#         fitList = []
#         for individual in population:
#             member = Individual.Individual()
#             cost = member.computeTotalCost(individual.chromosome)
#             fitList.append(member.getFitness(cost))
#             totalFit = sum(fitList)
#             u = np.random.uniform(0, totalFit)
#             current = 0
#         for j in np.arange(len(fitList)):
#             current += fitList[j]
#             if (current > u):
#                 return population[j]
        #print "fitness: ", population[0].fitness
        fitSum = sum([ind.fitness for ind in population])
        u = np.random.uniform(0, fitSum)
        current = 0
        for ind in population:
            current += ind.fitness
            if (current > u):
                return ind
                
    def tournamentSelection(self, population, candidateNum):
        candidatePopulation = [population[i] for i in np.random.randint(0, len(population), size = candidateNum)]
        candidatePopulation.sort(key = lambda x: x.cost)
        bestInd = candidatePopulation[0]
        return bestInd
            
    def getChild(self, population):            
#         if (self.selectionMethod == "rouletteWheel"):
#             parent1 = self.rouletteWheelSelection()
#             parent2 = self.rouletteWheelSelection()
#         elif (self.selectionMethod == "tournament"):
#             parent1 = self.tournamentSelection(self.population, candidateNum = self.populationSize/4)
#             parent2 = self.tournamentSelection(self.population, candidateNum = self.populationSize/4)
        parent1 = self.rouletteWheelSelection(population).chromosome
        parent2 = self.rouletteWheelSelection(population).chromosome
        childChromosome = Individual.Individual().uniformCrossover(parent1, parent2, randominzed = True)
        if (np.random.uniform() < self.probMutation):
            childChromosome = Individual.Individual().swapMutation(childChromosome)
        child = Individual.Individual(childChromosome)
        return child
    
    def updateGeneration(self, population, newGen):
        merged = population + newGen
        merged.sort(key = lambda x: x.cost)
        return merged[0:len(population)]
    
    def geneticAlgorithm(self):
        generation = 0
        replaceNum = int(self.populationSize * self.probCrossover)
        while (generation <= self.maxGeneration):
            newGen = []
            for i in np.arange(replaceNum):
                child = self.getChild(self.population)
                newGen.append(child)
            self.population = self.updateGeneration(self.population, newGen)
            generation += 1
        return self.population[0]
                
            
#===============================================================================
if __name__ == '__main__':
    init10 = TenInitialPopulations.TenInitialPopulations().init10
    psList = [50, 500, 2000]
    pmList = [.001, .5, .99]
    gmList = [20, 200, 1000]
    for ps in psList:
        print "############################################################"
        for i in np.arange(len(init10)):
            print "********************************************************"
            initPop = init10[i][0:ps]
            for gm in gmList:
                print "====================================================" 
                for pm in pmList:
                    print "------------------------------------------------"  
                    print "PopulationSize =", ps, ", Maximum Generation =", gm, ", Matation Probability =", pm, ", Initial Population Choice #", i+1
                    solveQAP = QAP_GA(probCrossover=.6, probMutation=pm, populationSize=ps, maxGeneration=100, initPopulation=initPop)
                    startTime = time.clock()
                    res = solveQAP.geneticAlgorithm()
                    print "Best Result Found: ", res.chromosome + 1
                    print "Minimum Total Cost Found: ", res.cost
                    endTime = time.clock()
                    print "Elapsed Time: ", endTime - startTime 