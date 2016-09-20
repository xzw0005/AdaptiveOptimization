'''
Created on Sep 15, 2016

@author: XING
'''
import numpy as np
import EvolutionaryStrategies.prod.Individual as Individual
from EvolutionaryStrategies.prod.AckleyFunction import ackley
import operator

class ES(object):
    '''
    Evolutionary Strategy for Continuous Function Optimization
    '''


    def __init__(self, mu, lam, sigma, population, maxiter = 5e3, strategy = "plus", recombination = False,
                 discrete_dual = False, global_intermediate = False, #intermediate_dual = False, global_discrete = False, 
                 bounds = [(-32.0, 32.0)] * 2, obj = "min"):
        '''
        strategy could be either "plus" or "comma":
        recombination only valid if strategy == "comma"
             could be "discrete_dual", or "intermediate_dual",
              or "global_discrete", or "global_intermediate"
            Only required to implement "discrete_dual" & "global_intermediate", would implement "intermediate_dual" & "global_discrete" later
        obj could be either "min" or "max"
        '''
        self.mu = mu
        self.lam = lam
        self.population = population
        self.maxiter = maxiter
        self.obj = obj
        self.approxResult = 1e-10
        self.bounds = bounds    
        
        self.strategy = strategy
        self.recombination = recombination
        if (global_intermediate==True or discrete_dual==True):
            self.recombination = True
        for individual in self.population:
            initSigma = np.zeros(len(bounds))
            if (global_intermediate == True):    # This Recombination Method most used for sigma
                initSigma += np.random.random(len(bounds)) * sigma * 2   # Randomly Initial Sigma for Individuals
            else:
                initSigma += sigma                               # Fixed Initial Sigma
            individual.sigma = initSigma#updateIndividualSigma(sigma)
            
        self.global_intermediate = global_intermediate
        self.discrete_dual= discrete_dual
        #self.intermediate_dual = intermediate_dual 
        #self.global_discrete = global_discrete
        
        self.generation = 0
        #self.n = 3  # 1/5 Success Rule applies every n generations, not mutations!!
        self.succ_nums = []   # 1/5 Rule based on success rate over last 10n mutations

    
    def oneFifthRule(self):
#         if (len(self.succ_nums) >= 10 * self.n):
#             succ_rate = sum(self.succ_nums) * 1.0 / len(self.succ_nums)
        if (self.generation > 0) and (self.generation % 10 == 0):
            succ_rate = sum(self.succ_nums) / (10 * self.lam)  #len(self.succ_nums)
            self.succ_nums = []
            if (succ_rate > 0.2):
                #print "~~~~~~~~~~~~~~~~~~~~~~~"
                self.updateSigma(proportion = 1 / 0.85)
                #sigma = sigma / 0.85
            else:
                #print "!!!!!!!!!!!!!!!!!!!!!!!"
                self.updateSigma(proportion = 0.85)
        pass

    def updateSigma(self, proportion, population = None):
        if (population is None):
            population = self.population
        for individual in population:
            individual.sigma *= proportion
        pass

    
#     def sortPopulation(self, population = None):
#         if (population is None):
#             population = self.population
#         if (self.obj == "min"):
#             population.sort(key = lambda ind: ackley(ind.xx))
#         elif (self.obj == "max"):
#             population.sort(key = lambda x : -x.fval)
#         pass
#         #return population
        
    def generateChild(self, population):
        parents = np.random.choice(population, size = 3, replace = False)
        if (self.recombination == True):
            child = self.recombinate(parents)
        else:
            xx = np.copy(parents[0].xx); sig = parents[0].sigma
            child = Individual.Individual(self.generation, xx, sig)
        return child.mutate()
    
    def recombinate(self, parents):
        assert len(parents) == 3
        p1 = parents[0]
        if (self.discrete_dual is True):
            rand_indices = np.random.randint(0, 2, size = len(p1.xx))
            new_xx = []; new_sigma = []
            for k in np.arange(len(p1.xx)):
                if rand_indices[k] == 0:
                    new_xx.append(p1.xx[k])
                    new_sigma.append(p1.sigma[k])
                else:
                    new_xx.append(parents[1].xx[k])
                    new_sigma.append(parents[1].sigma[k])
        elif (self.global_intermediate is True):
            new_xx = []; new_sigma = []
            for k in np.arange(len(p1.xx)):
                i, j = np.random.choice(np.arange(3), size = 2, replace=False)
                new_xx.append((parents[i].xx[k] + parents[j].xx[k])/2.0)
                new_sigma.append((parents[i].sigma[k] + parents[j].sigma[k])/2.0)
        return Individual.Individual(self.generation, np.array(new_xx), np.array(new_sigma))
        
    def shouldTerminate(self, population):
        return (self.generation > self.maxiter) or (self.getBest(population) < self.approxResult)
    
    def run(self):
        records = []
        #self.succ = 1
        self.fail = 1
        while not self.shouldTerminate(self.population):
            self.population, result = self.evolve(self.population)
            self.oneFifthRule()
            records.append(result)
        return records
    
    def evolve(self, population):
        children = [self.generateChild(population) for i in range(self.lam)]
        if (self.strategy == 'plus'):    
            newGen = self.population + children
        else:
            newGen = children
        fval_pairs = [(ind, ind.getfval()) for ind in newGen]
        new_pop_pairs = sorted(fval_pairs, key = operator.itemgetter(1))[:self.mu]
        new_population = [pair[0] for pair in new_pop_pairs]
        
        succ = [individual.gen == self.generation for individual in self.population]
        self.succ_nums.append(sum(succ))
        
        best_fval = self.getBest(new_population)
        mean_fval = np.mean([ind.getfval()[0] for ind in new_population])
        std_fval = np.std([ind.getfval()[0] for ind in new_population])
        self.generation += 1
        if self.generation % 100 == 0:
            print self.generation, new_population[0], best_fval
        return new_population, [self.generation, best_fval, mean_fval, std_fval]
                #population = self.sortPopulation(population)
            #else:       # otherwise, self.strategy would be 'comma'
        #children = self.sortPopulation(children)

        
#         if (population is None):
#             #population = np.copy(self.population)
#             population = self.population
#         if (self.global_intermediate is not True): #and (sigma is None):
#             sigma = self.sigma
            #raise ValueError("No individual sigma value, need a global sigma value!")
#          while not self.shouldTerminate(self.population):
#             children = []
#             for i in np.arange(self.mu):
#                 for j in np.arange(self.lam / self.mu):
#                     if (self.discrete_dual == True):
#                         for k in np.arange(len(parent1.xx)):
#                             if (np.random.uniform() < 0.5):
#                                 child_xx[k] = parent2.xx[k]
#                     child = Individual.Individual(self.generation, child_xx) 
#                     if (self.global_intermediate == True):
#                         parent3 = self.uniformSelect()
# #                         print '#1: ', parent1.sigma
# #                         print '#2: ', parent2.sigma
# #                         print '#3: ', parent3.sigma
#                         child.sigma = (parent1.sigma + parent2.sigma + parent3.sigma) / 3.0
#                         ###################child.updateIndividualSigma(sigma)
#                         #print 'child: ', sigma
#                     newkid = self.mutation(child)
#                     children.append(newkid)
#                     #####################if (self.global_intermediate is not True) and (len(children) % self.n == 0):
#             if (self.strategy == 'plus'):    
#                 children = self.population + children
#                 
#                     #population = self.sortPopulation(population)
#                 #else:       # otherwise, self.strategy would be 'comma'
#             #children = self.sortPopulation(children)
#             print children
#             children.sort(key = lambda x: ackley(x.xx))
#             print children
#             self.population = children[0:self.mu]
#             succ = [individual.gen == self.generation for individual in self.population]
#             self.succ_nums.append(sum(succ))
#             self.oneFifthRule()
#             print self.population[0].xx
#             print ackley(self.population[0].xx)
#             #print self.population
#             #print self.eval(self.population)
#             self.generation += 1
#         #return population
#     
    def getBest(self, population):
        return population[0].getfval()[0]