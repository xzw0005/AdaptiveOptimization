'''
Created on Sep 15, 2016

@author: Xing Wang
'''
import math

def ackley(xx):
    a = 20.0; b = 0.2; c = 2 * math.pi;
    sum1 = 0
    sum2 = 0
    for x in xx:
        sum1 += x**2
        sum2 += math.cos(c * x)
    n = float(len(xx))
    val = -a * math.exp(-b * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + a + math.e
    return val

# if __name__ == '__main__':
#     pass
# xx = np.array([13.76170006219458, 13.761700062194581])

#print ackley(xx)