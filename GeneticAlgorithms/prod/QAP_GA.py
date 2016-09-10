'''
Created on Sep 5, 2016
@author: Xing Wang
'''

import numpy as np
import time

import GeneticAlgorithms.prod.Individual as Individual
import GeneticAlgorithms.prod.Population as Population


class QAP_GA(object):
    '''
    Soving a Quadratic Assignment Problem of Department Locations, 
    by minimizing the total transportation through all departments.
    This is Q1 for HW2.
    '''

    def __init__(self, probCrossover, probMutation, populationSize, maxGeneration):
        '''
        Constructor
        '''
        self.populationSize = populationSize
        self.probCrossover = probCrossover
        self.probMutation = probMutation
        self.maxGeneration = maxGeneration
        self.generation = 0
        population = []
        while populationSize:
            member = Individual.Individual()
            #member.randomIndividual()
            population.append(member)
            populationSize -= 1
        self.population = population

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
# if __name__ == '__main__':
#     startTime = time.clock()
#     solveQAP = QAP_GA(T0 = 100, Tf = 1e-5, n = 500, alpha = .9, seed = 6542745252)
#     res = solveQAP.geneticAlgorithm()  
#     print "Best Result Found: ", res+1
#     print "Minimum Total Cost Found: ", solveQAP.computeTotalCost(res)
#     endTime = time.clock()
#     print "Elapsed Time: ", endTime - startTime     
#===============================================================================
# trycase = QAP_GA(probCrossover=.3, probMutation=.001, size = 100, maxGeneration = 10)
# pop, cl, fl = trycase.initializePopulation(5)
# print pop, cl, fl
# print "###", trycase.rouletteWheelSelection(pop)
# print "###", trycase.rouletteWheelSelection(pop)
# 
# p1 = np.random.permutation(15)
# p2 = np.random.permutation(15)
# print p1,"\n", p2
# print trycase.uniformCrossover(p1, p2, randominzed=True)
# print trycase.swapMutation(p1)


trycase = QAP_GA(probCrossover=.6, probMutation=.1, populationSize=1000, maxGeneration=200)
P = trycase.population
#print P
print trycase.geneticAlgorithm().cost