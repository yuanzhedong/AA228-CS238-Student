import pandas as pd
import numpy as np
import copy
import math

from functools import reduce
from decimal import Decimal
import itertools

from graphviz import Digraph
import pydot

raw_text='''
passengerclass,age
numsiblings,age
numparentschildren,age
passengerclass,portembarked
numsiblings,numparentschildren
survived,numparentschildren
fare,passengerclass
numsiblings,passengerclass
fare,sex
passengerclass,sex
numparentschildren,sex
survived,sex
fare,survived
passengerclass,survived
portembarked,survived
'''

def graph_from_dict(dictionary,):
    edge_style = ""
    g = Digraph(format='png')
   
    for k in dictionary.keys():
        if any([k in sub for sub in dictionary.values() for key in dictionary.keys()]) or dictionary[k]:
            g.node(str(k),k, shape='oval', fontsize='10', width='0', style='filled', fillcolor='#c9c9c9', color="gray") 

    for k, i in dictionary.items():
        for it in i:
            g.edge(str(it), str(k), label='',style= edge_style, color='black')  
    return g


lines = raw_text.split('\n')

import collections
graph = collections.defaultdict(list)

for line in lines:
    if len(line) > 0:
        p, c = line.split(',')
        graph[c].append(p)
g = graph_from_dict(graph)

g.render("/Users/nvidia/Downloads/large")
