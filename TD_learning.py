import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fractions import Fraction

alpha = 0.01
gamma = 0.99
num_states = 7
states = np.arange(num_states)
actions = ('A', 'B')
num_features = 15
num_runs = 10000
num_episodes = 500



phi_a = np.zeros((num_states, num_features))
phi_b = np.zeros((num_states, num_features))
w_0 = np.full(num_features, 1)
w_0[num_states-1] = 5


#transition probabilities for action A
p_a = np.ones((num_states,num_states))
for i in range(num_states-1):
	p_a[:,i] = 0

#transition probabilities for action B
p_b = np.zeros((num_states, num_states))
for i in range(num_states-1):
	p_b[:,i] = Fraction('1/6')
#print("Transition probabilities A", p_a)
#print("Transition probabilities B", p_b)


#phi_a
for i in range(num_states-1):
    phi_a[i,i] = 2
phi_a[num_states-1, num_states-1] = 1
phi_a[:,num_states] = 1
phi_a[-1,num_states] = 2


#phi_b
for i in range(1, num_states+1):
    phi_b[i-1, num_states+i] = 1 
    
phi = {}
phi['A'] = phi_a
phi['B'] = phi_b

#print("Parameter vector initialization A", phi_a)
#print("Parameter vector initialization B", phi_b)
#print("Parameter vectors initialization", w_0)


#System Dynamics
def transition(a, states = states):
    if a == 'A':
        s = states[-1]
    elif a == 'B':
        s = np.random.choice(states[:-1])
        #print(s)
    return s

def policy(actions = ('A','B'), p = [Fraction('1/7'), Fraction('6/7')]):
    a = np.random.choice(actions, p=p)
    #print(a)
    return a


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def _train_step_qlearning():
    # Initial approximation function of the q-value
    Q = {}
    for s in states:
        Q[s] = {}
        for a in actions:
            Q[s][a] = np.dot(w_0,phi[a][s])

    def update_Q(w, phi=phi, states=states, actions=actions):
        for s in states:
            Q[s] = {}
            for a in actions:
                Q[s][a] = np.dot(w,phi[a][s])
        return Q

    # print(Q)
    # number of the times the q-value has been updated
    # https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl/q_learning.py
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in actions:
            update_counts_sa[s][a] = 0
    
    # repeat until convergence
    norms = []
    
    
    w = w_0
    w_vec = np.empty((w.size, num_episodes+1))
    for i in range(w.size):
        w_vec[i,0] = w[i]
    norms.append(np.linalg.norm(w))
    a_0 = policy()
    s = transition(a_0)
    a = policy()
    for episode in range(num_episodes):
        
        s2 = transition(a)
        a2 = policy()
        
        old_w = w
        a_max, q_max = max_dict(Q[s2])
        w = old_w + alpha*(gamma*np.dot(w,phi[a_max][s2]) - np.dot(w,phi[a][s]))*phi[a][s]
        norms.append(np.linalg.norm(w))
        for i in range(w.size):
            w_vec[i,episode+1] = w[i]
        
        # check how often Q(s) has been updated too
        update_counts_sa[s][a] += 1
        update_counts[s] = update_counts.get(s,0) + 1
        
        Q = update_Q(w)
        
        s = s2
        a = a2
        
        #plt.plot(norms)
        #plt.show()
        
    policy_q = {}
    V = {}
    for s in states:
        a, max_q = max_dict(Q[s])
        policy_q[s] = a
        V[s] = max_q   
       
    return V, Q, policy_q, norms, w_vec

V, Q, policy_q, norms, w_vec = _train_step_qlearning()
for i in range(num_runs):
    V_n, Q_n, policy_n, norms_qlearning, w_vec_n = _train_step_qlearning()
    norms = np.multiply((i+1)/(i+2),norms) + np.multiply(1/(i+2),norms_qlearning)
    w_vec = np.multiply((i+1)/(i+2),w_vec) + np.multiply(1/(i+2),w_vec_n)
    
    for s in states:
        V[s] = (i+1)/(i+2)*V[s] + (1)/(i+2)*V_n[s]
        for a in actions:
            Q[s][a] = (i+1)/(i+2)*Q[s][a] + (1)/(i+2)*Q_n[s][a]

for s in states:
    a, max_q = max_dict(Q[s])
    policy_n[s] = a
    V[s] = max_q

policy_q = policy

def _train_step_sarsa():
    Q = {}
    for s in states:
        Q[s] = {}
        for a in actions:
            Q[s][a] = np.dot(w_0,phi[a][s])

    def update_Q(w, phi=phi, states=states, actions=actions):
        for s in states:
            Q[s] = {}
            for a in actions:
                Q[s][a] = np.dot(w,phi[a][s])
        return Q

    
    # number of the times the q-value has been updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in actions:
            update_counts_sa[s][a] = 0
    
    # repeat until convergence
    norms = []
    
    w = w_0
    norms.append(np.linalg.norm(w))
    a = policy()
    s = transition(a)
    for episode in range(num_episodes):
        
        a2 = policy()
        s2 = transition(a2)
        
        old_w = w
        w = old_w + alpha*(gamma*np.dot(w,phi[a2][s2]) - np.dot(w,phi[a][s]))*phi[a][s]
        norms.append(np.linalg.norm(w))
        
        # check how often Q(s) has been updated too
        update_counts_sa[s][a] += 1
        update_counts[s] = update_counts.get(s,0) + 1
        
        s = s2
        a = a2
    
    Q = update_Q(w)
    policy_sarsa = {}
    V = {}
    for s in states:
        a, max_q = max_dict(Q[s])
        policy_sarsa[s] = a
        V[s] = max_q
    
    return V, Q, policy_sarsa, norms

V, Q, policy_sarsa, norms = _train_step_sarsa()
for i in range(num_runs):
    V_n, Q_n, policy_n, norms_sarsa = _train_step_sarsa()
    norms = np.multiply((i+1)/(i+2),norms) + np.multiply(1/(i+2),norms_sarsa)
    
    for s in states:
        V[s] = (i+1)/(i+2)*V[s] + (1)/(i+2)*V_n[s]
        for a in actions:
            Q[s][a] = (i+1)/(i+2)*Q[s][a] + (1)/(i+2)*Q_n[s][a]

for s in states:
    a, max_q = max_dict(Q[s])
    policy_n[s] = a
    V[s] = max_q



print(norms_qlearning)
print(norms_sarsa)

plt.plot(norms_qlearning, color='blue', label='||w|| qlearning')
plt.plot(norms_sarsa, color='orange', label='||w|| sarsa')
plt.xlabel("Time steps")
plt.ylabel("||w||")
plt.legend() 
plt.show()
