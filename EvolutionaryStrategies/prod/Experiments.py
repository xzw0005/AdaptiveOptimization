'''
Created on Sep 18, 2016

@author: XING
'''
import numpy as np
import EvolutionaryStrategies.prod.InitPopulation as InitPopulation
import EvolutionaryStrategies.prod.Individual as Individual
import EvolutionaryStrategies.prod.ES as ES

import matplotlib
import matplotlib.pyplot as plt
from pylab import *

if __name__ == '__main__':
    mu = 10
    lam = 50
    sigma = 1  
        
    np.random.seed(123)
    tenSeeds = np.random.randint(0, 100, size=10)
    init10 = []
    for sd in tenSeeds:
        np.random.randint(sd)
        population = []
        for i in np.arange(mu):
            member = Individual.Individual()
            member.randomIndividual()
            population.append(member)
        init10.append(population)    

    ResList = []
    f = open("results_es.txt", 'w')
    
    fig = plt.figure()
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    mins = []
    
    for i in np.arange(10):
        es = ES.ES(mu, lam, sigma, init10[i], discrete_dual=True, global_intermediate=False, strategy="plus")
        records= es.run()
        x=[];y=[]
        for rec in records:
            f.write(str(rec) + '\n')
            x.append(rec[0])
            y.append(rec[2])
            #print rec[0], rec[2]
        mins.append(records[-1][3])
        plt.plot(x, y, c=colors[i])
        f.write("#############################################################\n")
    plt.xlim((0, 100))
    #print mins
    plt.ylim((0, 2)) 
    plt.xlabel("Generation")
    plt.ylabel("Minimum Ackley Function Value") 
    plt.title("Minimum Value in Each Generation, lambda/mu = 5, Discrete Dual")#, #Comma Strategy")  
    plt.show()
    f.close()