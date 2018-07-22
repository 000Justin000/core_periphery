from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pylab as plt
import random as rd
import time as t

start = t.time()

#compute the external field product, expressed as the sum of logs.
def compute_PROD2(N,phi,omega):
  PROD2 = [0]*2
  for r in range(2):
    for w in range(N):
      SIGMA = 0
      for s in range(2):
        SIGMA += phi[w][s]*np.exp(-omega[r][s])
      PROD2[r] += np.log(SIGMA)
  return PROD2
  
#compute the one-point marginal, probability a node belongs to each group, from message matrix.
def compute_phi(N,cav,phi,gamma,omega,PROD2,A):
  for i in range(N):
    BUFFER = [0]*2 
    for r in range(2):
      PROD = 0
      #compute the contribution from edges
      for j in A[i]:
        SIGMA1,SIGMA2 = 0,0
        for s in range(2):
          SIGMA1 += cav[j][i][s]*omega[r][s]*np.exp(-omega[r][s])
          SIGMA2 += phi[j][s]*np.exp(-omega[r][s])
        PROD += np.log(SIGMA1/SIGMA2)
      #compute unnormalized probability  
      BUFFER[r] = np.log(gamma[r]) + PROD + PROD2[r]
    #compute the ratio of two log probabilities    
    x = np.exp(BUFFER[0]-BUFFER[1])
    BUFFER = [1-1./(x+1),1./(x+1)] 
    #normalize so that the sum for each node is one
    phi[i] = [e/sum(BUFFER) for e in BUFFER] 
  return phi

#compute the message matrix
def compute_cav(N,cav,phi,gamma,omega,PROD2,A):
  for i in range(N):
    for j in range(N):
      if j in A[i]:    
        BUFFER = [0]*2
        for r in range(2):
          PROD = 0
          for k in A[i]:
            if k!=j:  
              SIGMA1,SIGMA2 = 0,0
              for s in range(2):
                SIGMA1 += cav[k][i][s]*(omega[r][s])*np.exp(-omega[r][s])
                SIGMA2 += phi[k][s]*np.exp(-omega[r][s])
              PROD += np.log(SIGMA1/SIGMA2)
            BUFFER[r] = np.log(gamma[r]) + PROD + PROD2[r]     
        #normalize
        x = np.exp(BUFFER[0] - BUFFER[1])
        BUFFER = [1-1./(x+1),1./(x+1)]
        cav[i][j] = [e/sum(BUFFER) for e in BUFFER]            
      else:
        cav[i][j] = phi[i]     
  return cav  

#compute the two-point marginal probability
def compute_jointmarginal(N,cav,omega,A):  
  joint_marg =np.zeros((N,N,2),dtype=object)
  for i in range(N):
    for j in range(N):
      if j!=i:
        Norm = 0
        BUFFER=np.zeros((2,2))
        for r in range(2):
          PROD = 0
          for s in range(2):
            if j in A[i]:
              PROD = omega[r][s]*np.exp(-omega[r][s])*cav[i][j][r]*cav[j][i][s]
              Norm += PROD
              BUFFER[r][s] = PROD              
            else:
              PROD = np.exp(-omega[r][s])*cav[i][j][r]*cav[j][i][s]
              Norm += PROD
              BUFFER[r][s]= PROD  
        for r in range(2):
          joint_marg[i][j][r]= [e/Norm for e in BUFFER[r]]  
  return joint_marg

#update the parameters using one-point and two-piont marginal probabilities
def update_parameters(N,phi,joint_marg,A):
  gamma,omega = np.zeros(2),np.zeros((2,2))
  #update gamma
  for r in range(2):
    SIGMA = 0
    for i in range(N):
      SIGMA += phi[i][r]
    gamma[r] = SIGMA
  gamma = [e/sum(gamma) for e in gamma]  
  #update omega
  for r in range(2):
    BUFFER = [0]*2
    for s in range(2):
      SIGMA = 0
      for i in range(N):
        for j in A[i]:
          SIGMA += joint_marg[i][j][r][s]
      BUFFER[s] = SIGMA/(gamma[r]*gamma[s]*N**2)    
    omega[r] = BUFFER
  return gamma,omega

#main belief propagation loop
def bp(N,gamma,omega,A,A1,itemax=50):
  #one could also change the stopping criteria to error tolerance.
  ite = 0  
  #initialization of the message matrix and the one-point marginal matrix.
  a,b = rd.random(),rd.random()
  a,b = 0.1,0.1
  cav = [a,1-a]*N*N
  cav = np.asarray(cav).reshape(N,N,2)
  phi = [b,1-b]*N
  phi = np.asarray(phi).reshape(N,2)
  while ite < itemax:
    #update messages  
    PROD2 = compute_PROD2(N,phi,omega)    
    cav = compute_cav(N,cav,phi,gamma,omega,PROD2,A)    
    phi = compute_phi(N,cav,phi,gamma,omega,PROD2,A)

    #update parameters
    joint_marg = compute_jointmarginal(N,cav,omega,A)
    gamma,omega = update_parameters(N,phi,joint_marg,A)
    print omega

    #assigning nodes to groups with higher probability.
    assign = [list(e).index(max(e)) for e in phi]      
    ite += 1      
  return assign,phi,gamma,omega

def main():
  G = nx.read_gml('karate.gml')
  G = nx.convert_node_labels_to_integers(G)
  n = len(G.nodes())
  #initial guesses for the parameters, may influence the point of convergence, try different initialization
  gamma =[0.5, 0.5]
  omega =np.array([[ 0.5 , 0.3],[ 0.3  ,0.1]])
  A = G.adjacency_list()
  A1 = nx.adjacency_matrix(G)
  assign,phi,gamma,omega = bp(n,gamma,omega,A,A1)  
 
  plt.clf()
  nx.draw(G,with_labels=False,node_color = assign)
  plt.show()
  return assign,phi,gamma,omega
  
if __name__=="__main__":
  assign,phi,gamma,omega = main()
  print str(t.time() - start ) + " time"
