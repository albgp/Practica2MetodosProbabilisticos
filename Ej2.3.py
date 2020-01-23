#YET TO COMMENT.

import numpy as np
from functools import reduce

class ProbabilityNetwork:
    def __init__(self,n,edges,probs):
        self.nodes=list(range(n))
        self.edges=edges
        self.probs=probs

    def parents(self, node):
        return [a for a,b in edges if b==node]

    def ancestralOrder(self):
        order=[]
        while len(order)<len(self.nodes):
            for node in self.nodes:
                if node in order:
                    continue
                if not any((edge[0] not in order) and (edge[1]==node) for edge in self.edges):
                    order.append(node)
        return order

    def logicSampling(self, evidences, targetNode, niters=10000000):
        evidenceNodes=evidences.keys()
        ancestralOrder = self.ancestralOrder()
        hits=0
        total=0

        for it in range(niters):
            fail=False
            values=dict([ [i,None] for i in self.nodes]) #True: present. False: not present
            for node in ancestralOrder:
                pNode=self.probs(node, values)
                nodeValue=np.random.random()<pNode
                values[node]=nodeValue
                if node in evidences and evidences[node]!=values[node]:
                    fail=True
                    break

            if fail: continue

            #print(values)
            total+=1
            if values[targetNode]:
                hits+=1

        return hits/total

    def weightedLikelihood(self, evidences, targetNode, niters=10000000):
        evidenceNodes=evidences.keys()

        ancestralOrder = [node for node in self.ancestralOrder() if node not in evidenceNodes]
        cumsumHit=0
        cumsumTotal=0
        hits=0
        for it in range(niters):
            values=dict([ [i,None] for i in ancestralOrder]) #True: present. False: not present
            for evNode in evidenceNodes:
                values[evNode]=evidences[evNode]

            for node in ancestralOrder:
                pNode=self.probs(node, values)
                nodeValue=np.random.random()<pNode
                values[node]=nodeValue

            currProb=reduce(lambda x,y:x*y, [self.probs(i,values) if values[i] else 1-self.probs(i,values) for i in evidenceNodes ])
            if values[targetNode]:
                cumsumHit+=currProb

            cumsumTotal+=currProb

        return cumsumHit/cumsumTotal



edges=[(0,1),(0,2),(1,3),(1,4),(2,4),(2,5)]

def probs(node,evidences):
    if node==0: return 0.3
    elif node==1:
        if evidences[0]: return 0.9
        else: return 0.2
    elif node==2:
        if evidences[0]: return 0.75
        else: return 0.25
    elif node==3:
        if evidences[1]: return 0.6
        else: return 0.1
    elif node==4:
        if evidences[1] and evidences[2]: return 0.8
        elif evidences[1] and not evidences[2]: return 0.6
        elif not evidences[1] and evidences[2]: return 0.5
        else: return 0
    elif node==5:
        if evidences[2]: return 0.4
        else: return 0.1

pn=ProbabilityNetwork(6, edges, probs)

evidences=dict([[3,True],[4,True],[5,False]])

print(pn.logicSampling(evidences, 0))
print(pn.weightedLikelihood(evidences,0))



