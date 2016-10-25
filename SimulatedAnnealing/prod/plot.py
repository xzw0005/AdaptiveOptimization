'''
Created on Oct 16, 2016

@author: XING
'''
length = [1, 2, 3, 4]
xtick_label = [20, 50, 100, 1000]
RHC = [20.0/20, 50.0/50, 100.0/100, 1000.0/1000]
SA = [20.0/20, 50.0/50, 100.0/100, 1000.0/1000]
GA = [16.0/20, 43.0/50, 76.0/100, 608.0/1000]
MIMIC = [18.0/20, 42.0/50, 72.0/100, 698.0/1000]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(length, SA, 'r', label='SA')
ax.plot(length, GA, 'b', label='GA')
ax.plot(length, MIMIC, 'g', label = 'MIMIC')
ax.legend(loc='lower left', shadow=True)
plt.ylim(0.4, 1.01)
plt.xticks(length, xtick_label)
yvals=ax.get_yticks()[:-1]
ys = ['{:.1f}%'.format(y*100) for y in yvals]
plt.xticks(length, xtick_label)
plt.yticks(yvals, ys)
#ax.set_yticklabels(['{:.1f%}%'.format(x*100) for x in yvals])
plt.show()