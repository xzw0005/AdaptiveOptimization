'''
Created on Sep 16, 2016

@author: XING
'''
import EvolutionaryStrategies.prod.Individual as Individual
import numpy as np
import numbers

class InitPopulation(object):
    
    def __init__(self, seed, popsize, bounds = [(-32, 32), (-32, 32)]):
        #self.init10 = []
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence of (min, max) pairs for each value in xx')        
        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = self.check_random_state(seed)
        
        self.popsize = popsize  
        #self.num_population_members = self.popsize * self.parameter_count
        self.num_population_members = self.popsize
        self.popshape = (self.num_population_members, self.parameter_count)

        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = abs(self.limits[0] - self.limits[1])    
        
        xx_mat = self.initPopLhs()    
        self.population = []
        for xx in xx_mat:
            ind = Individual.Individual(0, xx)
            self.population.append(ind)
     
    def initPopRandom(self):
        rng = self.random_number_generator      # rng is a container with seed, e.g., rng.random_sample() is just as np.random.seed(); np.random.random_sample()        
        #self.population = rng.random_sample(self.popshape)
        sample = rng.random_sample(self.popshape)
        xx_mat = self.__scale_arg1 + (sample - 0.5) * self.__scale_arg2 
        return xx_mat
        
    def initPopLhs(self):
        """
        Initialize the population with Latin Hypercube Sampling
        LHS ensures that each parameter is uniformly sampled over its range
        """
        rng = self.random_number_generator      # rng is a container with seed, e.g., rng.random_sample() is just as np.random.seed(); np.random.random_sample()
        segsize = 1.0 / self.popsize
        samples = (segsize * rng.random_sample(self.popshape) + np.linspace(0., 1., self.num_population_members, endpoint=False)[:, np.newaxis])
        samples = self.scaleParameters(samples)        
        xx_mat = np.zeros_like(samples)
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))    
            xx_mat[:, j] = samples[order, j]
        return xx_mat
            
    def scaleParameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        
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
    
# popu = InitPopulation(1, 20)
# popu.initPopLhs()
# print popu.population
# print popu.popshape