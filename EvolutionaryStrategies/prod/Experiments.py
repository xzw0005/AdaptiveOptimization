'''
Created on Sep 18, 2016

@author: XING
'''
import numpy as np
import EvolutionaryStrategies.prod.InitPopulation as InitPopulation
import EvolutionaryStrategies.prod.Individual as Individual
import EvolutionaryStrategies.prod.ES as ES

if __name__ == '__main__':
    mu = 20
    lam = 80
    sigma = 1  
        
    np.random.seed(12345)
    tenSeeds = np.random.randint(0, 100, size=10)
    init10 = []
    for sd in tenSeeds:
        np.random.randint(sd)
        indivSeeds = np.random.randint(0, 1e6, size = mu)
        population = []
        for seed in indivSeeds:
            member = Individual.Individual()
            member.randomIndividual()
            population.append(member)   
        init10.append(population)
    #for pop_i in init10:
    #print init10[9]
    es = ES.ES(mu, lam, sigma, init10[5], discrete_dual=False, global_intermediate=True)
    #print es.sortPopulation()
    print es.run()
        
#         print es.population
#         print es.eval(es.population)
#         
#     es = ES.ES(mu, lam, sigma, init10[5])
#     print es.population
#     print es.eval(es.population)