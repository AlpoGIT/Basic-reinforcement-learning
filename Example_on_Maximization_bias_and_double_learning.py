'''
("Reinforcement learning", Sutton, example on Maximization bias and double learning)
Plots the % left actions from state 2 (figure 6.5 of the book)
States [0,1,2,3]
Initial state: 2
Terminal states: 0 and 3
Description from the book:
"The MDP has two non-terminal states A and B. Episodes always start in A with a choice between two actions, left and right.
The right action transitions immediately to the terminal state with a reward and return of zero.
The left action transitions to B, also with a reward of zero, from which there are many
possible actions all of which cause immediate termination with a reward drawn from a
normal distribution with mean −0.1 and variance 1.0."
'''

import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.1
alpha = 0.1
gamma = 1.

nb_fork = 10 #possible actions leading to terminal state 0 from state 1
bias = -.1

n_average = 500   #10000 in the book
left_dQ = np.zeros(300)
mean_dQ = np.zeros(300)
left_Q = np.zeros(300)
mean_Q = np.zeros(300)

def rargmax(Q1,Q2,state): #returns a uniform random argmax
    temp = Q1[state]+Q2[state]
    action = np.random.choice(np.flatnonzero(temp == temp.max()))
    return action

def action_espilon_greedy(Q1,Q2,state,epsilon): #returns espilon-greedy action
    if state == 1:
        action = rargmax(Q1,Q2,state)
        return action

    rd = np.random.rand()
    if rd < epsilon:
        action = np.random.choice(np.arange(len(Q1[state])))
    else:
        action = rargmax(Q1,Q2,state)
              
    return action

def env(action, state):
    if state == 1:
        reward = np.random.normal(bias,1)
        state = 0
        return reward, state
    elif state == 2:
        state = int(state + 2*action -1)
        reward = 0

    return reward, state

#Main loop for Q and double Q learning
#loop for averaging
for j in range(n_average):  
    if j%100 == 0:
        print("\r{}/{}".format(j,n_average), end = "")
     
    #definition of Q matrix
    Q = {0:np.zeros(1), 1 : np.zeros(nb_fork), 2 : np.zeros(2), 3 : np.zeros(1)} 
    Q1 = {0:np.zeros(1), 1 : np.zeros(nb_fork), 2 : np.zeros(2), 3 : np.zeros(1)} 
    Q2 = {0:np.zeros(1), 1 : np.zeros(nb_fork), 2 : np.zeros(2), 3 : np.zeros(1)} 
    
    #loop on episodes
    for i in range(300):
        #########################
        ### double Q-learning
        #########################
        counter_left_dQ = 0
        total_action_dQ= 0
        state = 2 #initial state
        
        while True:
            #choose A from S using epsilon-greedy in Q1+Q2
            action = action_espilon_greedy(Q1,Q2,state,epsilon)
            reward, next_state = env(action, state)
            
            #some statistics
            if state == 2:
                total_action_dQ += 1
            if (action == 0) & (state == 2):
                counter_left_dQ += 1
                #print("dQ(state, action) = ({},{})".format(state,action))

            rd = np.random.rand()

            if rd < 0.5:
                Q1[state][action] += alpha*(reward + \
                    gamma*Q2[next_state][rargmax(Q1,Q1,next_state)]\
                    -Q1[state][action])
            else:
                Q2[state][action] += alpha*(reward + \
                    gamma*Q1[next_state][rargmax(Q2,Q2,next_state)]\
                    -Q2[state][action])

            state = next_state
            
            if state in [0,3]: #terminal states
                left_dQ[i] = counter_left_dQ/total_action_dQ # % left actions ith episode
                break
  
        #########################
        ### Q-learning
        #########################
        counter_left_Q = 0
        total_action_Q = 0
        state = 2 #initial state
        
        while True:
            #choose A from S using epsilon-greedy in Q
            action = action_espilon_greedy(Q,Q,state,epsilon)
            reward, next_state = env(action, state)
            
            #some statistics
            if state == 2:
                total_action_Q += 1
            if (action == 0) & (state == 2):
                counter_left_Q += 1
           
            Q[state][action] += alpha*(reward + gamma*np.max(Q[next_state])\
                        -Q[state][action])

            state = next_state
            if state in [0,3]: #terminal states
                left_Q[i] = counter_left_Q/total_action_Q # % left actions ith episode
                break
    
    mean_dQ = mean_dQ + left_dQ
    mean_Q = mean_Q + left_Q
          
mean_dQ = mean_dQ/n_average
mean_Q = mean_Q/n_average

plt.title("Maximization Bias and Double Learning")
plt.plot(mean_dQ, label = 'Double Q-learning')
plt.plot(mean_Q, label = 'Q-learning')
plt.xlabel('Episodes')
plt.ylabel('% left actions')
plt.legend()
plt.show()
