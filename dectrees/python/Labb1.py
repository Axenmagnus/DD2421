#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:14:15 2022

@author: magnusaxen
"""
import matplotlib.pyplot as plt
import monkdata as m
from dtree import entropy
from dtree import averageGain
from dtree import buildTree
from dtree import select
from dtree import mostCommon
import numpy
Entropy1=entropy(m.monk1)
Entropy2=entropy(m.monk2)
Entropy3=entropy(m.monk3)

Entropies=[]

Monklist=[m.monk1,m.monk2,m.monk3]
Attributelist=["A1","A2","A3","A4","A5","A6"]
Vallist=[[1,2,3],[1,2,3],[1,2],[1,2,3],[1,2,3,4],[1,2]]

Table= numpy.zeros(shape=(3,6))
i=0

for monk in Monklist:
    j=0
    for att in Attributelist:
        Table[i,j]=averageGain(monk,m.attributes[j])
        j+=1

    i+=1
        


#print(Table)
#print(sum(Table))

InitalEntropy=entropy(m.monk1)

subset1=select(m.monk1,m.attributes[4],Vallist[4][0])
subset2=select(m.monk1,m.attributes[4],Vallist[4][1])
subset3=select(m.monk1,m.attributes[4],Vallist[4][2])
subset4=select(m.monk1,m.attributes[4],Vallist[4][3])
Table2 = numpy.zeros(shape=(4,6))

subsetslist=[subset1,subset2,subset3,subset4]

i=0
for subset in subsetslist:
    j=0
    for att in Attributelist:
        Table2[i,j]=averageGain(subset,m.attributes[j])
        j+=1
    i+=1


print(Table2)
print(sum(Table2))

#A1 should be tested for the next nodes.


import dtree as d
import random
from dtree import allPruned
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def proon(fraction,dataset):
    Bool=True
    cap=0
    monk1train, monk1val = partition(dataset, fraction)
    t=d.buildTree(monk1train, m.attributes);
    BestPrune=999999999 #it's over 9000!
    while Bool==True:
        
        if cap==0:
            Pruned=allPruned(t)
        else:
            Pruned=allPruned(BestPrune)
        val=0
        for prune in Pruned:
    
    
    
            
            if d.check(prune, monk1val)>val:
                val=d.check(prune, monk1val)
                best=val
                BestPrune=prune
                
        #print("best",best)
        if (cap>best):
            break
        
        cap=best
    return cap
    
    
    
Fractions=[i/10 for i in range(3,9)]
runs=100
performance=numpy.zeros(shape=(1,6))
k=0
for fraction in Fractions:
    
    i=0
    errorsum=0
    while runs>i:
        
        accuracy=proon(fraction,m.monk1)
        error=1-accuracy
        errorsum=errorsum+error
        i=i+1

    averror=errorsum/i
    performance[0,k]=averror
    k+=1



#print(performance)



