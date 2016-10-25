'''
Created on Oct 25, 2016

@author: Xing Wang
'''
import numpy as np
import numbers
import copy
import sys

def evaluate(position):
    a = 20.0; b = 0.2; c = 2 * np.pi;
    sum1 = 0.0
    sum2 = 0.0
    for xi in position:
        sum1 += xi**2
        sum2 += np.cos(c * xi)
    n = len(position)
    val = -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e
    return val

class Particle(object):
    def __init__(self, xmax, vmax, dimension, seed):
        if seed is None:
            self.rng = np.random.mtrand._rand
        elif isinstance(seed, (numbers.Integral, np.integer)):
            self.rng = np.random.RandomState(seed)
        else:
            raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)
        self.position = [-xmax + (xmax*2.)*self.rng.uniform() for i in range(dimension)]
        self.velocity = [-vmax + (vmax*2.)*self.rng.uniform() for i in range(dimension)]
        self.fval = evaluate(self.position)
        self.pbest_pos = copy.copy(self.position)
        self.pbest_fval = self.fval
       
        
def pso(swarm_size=30, maxiter=1e4, xmax=5., vmax=3., dimension=2, phi_cognition=2., phi_social=2., inertia=.5, tol=1e-10):
    rng = np.random.RandomState(6543210)
    swarm = [Particle(xmax, vmax, dimension, i) for i in np.arange(swarm_size)]
    gbest_pos = None
    gbest_fval = sys.float_info.max 
    for particle in swarm:
        if (particle.fval < gbest_fval):
            gbest_fval = particle.fval
            gbest_pos = copy.copy(particle.position)
    iter = 0
    results = []
    while (iter < maxiter) and (gbest_fval > tol):
        tot = 0
        for particle in swarm:
            for d in range(dimension):
                particle.velocity[d] =  inertia * particle.velocity[d] + \
                                            phi_cognition * rng.uniform() * (particle.pbest_pos[d] - particle.position[d]) + \
                                                phi_social * rng.uniform() * (gbest_pos[d] - particle.position[d]) 
                if (particle.velocity[d] < -vmax):
                    particle.velocity[d] = -vmax
                elif (particle.velocity[d] > vmax):
                    particle.velocity[d] = vmax
            
                particle.position[d] += particle.velocity[d]
            
            particle.fval = evaluate(particle.position)
            tot += particle.fval
            if (particle.fval < particle.pbest_fval):
                particle.pbest_fval = particle.fval
                particle.pbest_pos = copy.copy(particle.position)
            if (particle.fval < gbest_fval):
                gbest_fval = particle.fval
                gbest_pos = copy.copy(particle.position)
        mean_fval = tot / swarm_size
        results.append([iter, gbest_pos, gbest_fval, mean_fval])
        iter += 1
    return gbest_pos, gbest_fval, results
                    
                    
print pso()
                    
                    
                    
                    