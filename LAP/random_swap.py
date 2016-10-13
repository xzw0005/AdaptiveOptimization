'''
Created on Oct 11, 2016

@author: Xing Wang
'''

import LAP.lap as lap

fname = 'example1.csv'
costs, people, tasks = lap.initialize(fname, show=0)
print costs