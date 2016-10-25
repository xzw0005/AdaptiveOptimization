'''
Created on Sep 29, 2016

@author: XING WANG
'''

import numpy as np
import numbers
import time

import matplotlib
import matplotlib.pyplot as plt
from pylab import *

class DifferentialEvolutionOptimizer(object):
    '''
    classdocs
    '''


    def __init__(self, popsize=50, f=0.5, cr=0.8, x='rand', y=1, z='bin', seed = 123,
                 maxiter = 1e2, tol=1e-5, bounds = [(-5, 5), (-5, 5)]):
        '''
        Constructor
        '''
        self.popsize = popsize
        self.f = f
        self.cr = cr
        self.maxiter = maxiter
        self.tol = tol
        self.x = x
        self.y = y
        self.z = z
        self.population = []
        self.generation = 0

        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence of (min, max) pairs for each value in xx')        
        self.dimension = np.size(self.limits, 1)
        self.random_number_generator = self.check_random_state(seed)
        self.popshape = (self.popsize, self.dimension)

    def init_pop_random(self):
        rng = self.random_number_generator      # rng is a container with seed, e.g., rng.random_sample() is just as np.random.seed(); np.random.random_sample()        
        #self.population = rng.random_sample(self.popshape)
        sample = rng.random_sample(self.popshape)
        self.population = 0.5 * (self.limits[0] + self.limits[1]) + (sample - 0.5) * abs(self.limits[0] - self.limits[1])
        self.population = sorted(self.population, key = lambda individual : self.fval(individual) ) 
        
    # copy-pasted from scikit-learn utils/validation.py
    # Note that np.random.RandomState is a RNG container with a parameter "seed"
    def check_random_state(self, seed):
        """Turn seed into a np.random.RandomState instance
        If seed is None (or np.random), return the RandomState singleton used
        by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
        """
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)
        
    
    def fval(self, xx):         # ackley function
        a = 20.0; b = 0.2; c = 2 * np.pi;
        sum1 = 0
        sum2 = 0
        for x in xx:
            sum1 += x**2
            sum2 += np.cos(c * x)
        n = float(len(xx))
        val = -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e
        return val
    
    def mutate(self, i):
        idxs = range(self.popsize)
        idxs.remove(i)
        np.random.shuffle(idxs)
        if (self.x == 'rand') and (self.y == 1):    # rand/1/
            r0, r1, r2 = idxs[:3]
            mutant = self.population[r0] + self.f * (self.population[r1] - self.population[r2])            
        elif (self.x == 'best') and (self.y == 1):  # best/1/
            #self.population = sorted(self.population, key = lambda idx : self.fval(self.population[idx]) )
            r1, r2 = idxs[:2]
            mutant = self.population[0] + self.f *  (self.population[r1] - self.population[r2])
        elif (self.x == 'rand') and (self.y == 2):  # rand/2/
            r0, r1, r2, r3, r4 = idxs[:5]
            mutant = self.population[r0] + self.f * (self.population[r1] - self.population[r2] + self.population[r3] - self.population[r4])
        elif (self.x == 'best') and (self.y == 2):  # rand/2/
            r1, r2, r3, r4 = idxs[:4]
            mutant = self.population[0] + self.f * (self.population[r1] - self.population[r2]
                                                     + self.population[r3] - self.population[r4])
        elif (self.x == 'rand to best') and (self.y == 1):  # rand to best/2/
            r0, r1, r2 = idxs[:3]
            mutant = self.population[r0] + self.f * (self.population[r1] - self.population[r2]
                                                     + self.population[0] - self.population[r0])
        elif (self.x == 'current to best') and (self.y == 1):  # rand to best/2/
            r1, r2 = idxs[:2]
            mutant = self.population[i] + self.f * (self.population[r1] - self.population[r2]
                                                     + self.population[0] - self.population[i])
        return mutant
    
    def crossover(self, target, mutant, i):
        trial = np.copy(target)
        if (self.z == 'bin'): 
            for j in np.arange(self.dimension):
                if (np.random.rand() <= self.cr) or (j == i):
                    trial[j] = mutant[j]
        return trial
        
    def evolve(self):
        self.init_pop_random()
        results = []
        while(self.generation < self.maxiter and self.fval(self.population[0]) > self.tol):
            children = []
            for i in np.arange(self.popsize):
                mutant = self.mutate(i)
                target = self.population[i]
                trial = self.crossover(target, mutant, i)
                children.append(trial)
            for i in np.arange(self.popsize):
                if self.fval(children[i]) < self.fval(self.population[i]):
                    self.population[i] = np.copy(children[i])
            self.generation += 1
            self.population = sorted(self.population, key = lambda individual : self.fval(individual) )
            #print (self.population[0], self.fval(self.population[0]), self.generation)
            best_fval = self.fval(self.population[0])
            if self.generation == 50:
                f2.write(str(best_fval)+'\t')
            mean_fval = np.mean([self.fval(individual) for individual in self.population])
            results.append([self.generation, self.population[0], best_fval, mean_fval])
        return results


popsize=20; F=0.5; CR=0.9; X='rand'; Y=1; Z='bin'; maxiter = 500; tol=1e-5;
np.random.seed(123)
ten_seeds = np.random.randint(0, 1e3, size=10)
print ten_seeds
fh = open('F%dCR%d%s%d%s.txt'%(F*10, CR*10, X, Y, Z), 'w')
fig = plt.figure()
colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
mins = []
f2 = open('%s%d.txt'%(X, Y), 'w')
for i in range(len(ten_seeds)):
    print '---------------------#%d: seed=%d---------------------------'%(i+1,ten_seeds[i])
    fh.write('---------------------#%d: seed=%d---------------------------\n'%(i+1,ten_seeds[i]))
    ex = DifferentialEvolutionOptimizer(popsize=popsize, f=F, cr=CR, x=X, y=Y, z=Z, maxiter=maxiter, tol=tol, seed = ten_seeds[i])
    records = ex.evolve()
    x=[];y=[]
    for rec in records:
        print rec
        fh.write(str(rec) + '\n')
        x.append(rec[0])
        y.append(rec[2])
        #print rec[0], rec[2]
    mins.append(records[-1][3])
    plt.plot(x, y, c=colors[i])
plt.xlim((0, 50))
#print mins
plt.ylim((0, 5)) 
plt.xlabel("Iteration")
plt.ylabel("Minimum Ackley Function Value") 
plt.title("Minimum Value in Each Iteration")#, F=%.1f, CR=%.1f, %s/%d/%s" % (F, CR, X, Y, Z) )#, #Comma Strategy") 
#plt.savefig('F%dCR%d%s%d%s.png'%(F*10, CR*10, X, Y, Z)) 
plt.show()
fh.close()
f2.close()