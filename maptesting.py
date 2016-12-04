# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:06:28 2016

@author: Shummie
"""

import numpy
width = 20
height = 20

a = numpy.zeros((width, height))


a[10, 10] = 5
a[10, 9] = 4
a[9, 10] = 4
a[9, 8] = 7
a[9, 7] = 5
a[8, 6] = 3
a[8, 7] = 4
a[8, 8] = 2
a[7, 5] = 6

print(a)


import copy
import math

b = copy.copy(a)
print(b)

c = numpy.zeros((width, height))

decay = 0.2


def distance_between(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if dx > width / 2:
        dx = width - dx
    if dy > height / 2:
        dy = height - dy
    return dx + dy

for x in range(0, 20):
    for y in range(0, 20):
        if a[x,y] != 0: # No need to waste resources propogating 0
            for i in range(0, 20):
                for j in range(0, 20):
                    distance = distance_between(x, y, i, j)
                    c[i, j] += a[x,y] * math.exp(-decay * distance)
                    
print (c)                    