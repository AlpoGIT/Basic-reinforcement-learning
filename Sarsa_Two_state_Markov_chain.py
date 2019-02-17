import numpy as np
from collections import defaultdict
Q = defaultdict(lambda : np.zeros(1))

epsilon = 0.05
alpha = 0.01
gamma = 1.
reward = 1.

def env(state):
    next_state = np.random.choice([0,1],p=[0.75,0.25])
    reward = 1 if state == 0 else 0
    #print(reward, next_state)
    return reward, next_state

def epsilon_greedy(Q, state, epsilon):
    rnd = np.random.rand()
    if rnd < epsilon:
        action = np.random.choice([0])#only one action
    else:
        action = np.argmax(Q[state])
    return action

for i in range(50000):
    #epsilon = epsilon*0.999
    state = 0
    action = epsilon_greedy(Q,state,epsilon)
    
    while True:
        reward, next_state = env(state)
        next_action = epsilon_greedy(Q,next_state,epsilon)
        Q[state][action] += alpha*(reward + gamma*Q[next_state][next_action]-\
            Q[state][action])
        state = next_state
        if state == 1:
            break
print(dict(Q))



