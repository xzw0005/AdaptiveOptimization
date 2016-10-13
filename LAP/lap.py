'''
Created on Oct 11, 2016
@author: Xing Wang
'''

import sys
import copy

def readCSV(fname, delimiter=',', data_type='float', comment_char='%', 
            jagged=False, fill_value=0):
    matrix = []
    with open(fname, 'U') as f:
        lines = f.readlines()
        max_size = 0
    for line in lines:
        if (line.startswith(comment_char)):
            continue
        if line.rstrip():
            if data_type == 'float':
                row = map(float, filter(None, line.rstrip().split(delimiter)))
            elif data_type == 'int':
                row = map(int, filter(None, line.rstrip().split(delimiter)))
            matrix.append(row)
            if (len(row) > max_size):
                max_size = len(row)
    if not jagged:
        for row in matrix:
            row += [fill_value] * (max_size - len(row))
    if len(matrix) == 1:
        matrix = matrix[0]
    return matrix
    
def initialize(fname, show):
    costs = readCSV(fname)
    people = []
    tasks = []
    for i in range(len(costs)):
        people.append(-1)
        tasks.append(-1)
    return costs, people, tasks

def show_solution(costs, people, tasks, show_costs = 0):
    if show_costs:
        print 'Costs: ', costs
    print 'People: ', people
    print 'Tasks: ', tasks
    total_cost, num_assigned = calc_cost(costs, people)
    print 'Total Cost: ', total_cost
    print 'Number Assigned: ', num_assigned
    
def low_cost_task(costs, person, tasks):
    min_cost = sys.maxint
    min_idx = -1
    for k in range(len(tasks)):
        if (tasks[k] == -1):
            if (costs[person][k] < min_cost):
                min_cost = costs[person][k]
                min_idx = k
    return min_idx, min_cost
                
def calc_cost(costs, people):
    total_cost = 0
    num_assigned = 0
    for k in range(len(people)):
        if (people[k] != -1):
            total_cost += costs[k][people[k]]
            num_assigned += 1
    return total_cost, num_assigned

def simple_lower_bound(costs):
    tc1 = 0
    for k in range(len(costs)):
        tc1 += min(costs[k])
    tc2 = 0
    for k in range(len(costs)):
        tc2 += min([c[k] for c in costs])
    return max(tc1, tc2)

def store_solution(people, tasks, cost, seq):
    solution = {}
    solution['people'] = copy.copy(people)
    solution['tasks'] = copy.copy(tasks)
    solution['obj_val'] = cost
    solution['seq'] = copy.copy(seq)
    return solution

def clear_solution(people, tasks):
    for i in range(len(people)):
        people[i] = -1
    for i in range(len(tasks)):
        tasks[i] = -1
        
        