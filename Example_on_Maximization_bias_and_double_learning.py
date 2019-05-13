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
normal distribution with mean -0.1 and variance 1.0."
'''
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class Q_estimator():
    def __init__(self):
        self.nb_actions = 10
        self.value = {
                    0 : np.zeros(1),
                    1 : np.zeros(self.nb_actions),
                    2 : np.zeros(2),
                    3 : np.zeros(1)
                    }
        
class policy():
    def __init__(self):
        None
    def act(self, state, Q, params):
        rd = np.random.uniform(0,1)
        if rd < params['epsilon']:
            action = np.random.choice(np.arange(len(Q[0].value[state])))
        else:
            action = np.random.choice(np.flatnonzero( Q[0].value[state]+Q[1].value[state]==np.max(Q[0].value[state]+Q[1].value[state]) ))
        return action

class environment():
    def __init__(self):
        None

    def reset(self):
        state = 2
        return state

    def step(self, state, action):
        if state == 2:
            next_state = state + (2*action-1)
            reward = 0
            done = True if next_state == 3 else False
        elif state == 1:
            next_state = 0
            reward = np.random.normal(-0.1,1)
            done = True
        return next_state, reward, done

class class_stats():
    def __init__(self):
        self.max_ep = 300
        self.nb_left_action = np.zeros(self.max_ep)
        self.total = np.zeros(self.max_ep)

params = {
    'alpha' : 0.1,
    'epsilon' : 0.1,
    'gamma' : 1
    }

env = environment()
policy = policy()
stats = class_stats()

max_ep = 300
max_k = 1000

# double Q-learning
for k in range(max_k):
    Q1 = Q_estimator()
    Q2 = Q_estimator()
    Q = [Q1, Q2]

    for i in range(max_ep):
        state = env.reset()
        while True:
            action = policy.act(state, Q, params)
            next_state, reward, done = env.step(state, action)

            rd = np.random.uniform(0,1)

            if rd < 0.5:
                delta = reward+params['gamma']*Q[1].value[next_state][np.argmax(Q[0].value[next_state])]
                Q[0].value[state][action] += params['alpha']*(delta-Q[0].value[state][action])
            else:
                delta = reward+params['gamma']*Q[0].value[next_state][np.argmax(Q[1].value[next_state])]
                Q[1].value[state][action] += params['alpha']*(delta-Q[1].value[state][action])
            

            if action == 0 and state == 2:
                stats.nb_left_action[i] += 1
            if state == 2:
                stats.total[i] += 1


            state = next_state
            if done:
                break
# plot results
percentage = np.array([stats.nb_left_action[i]/ stats.total[i] for i in range(max_ep)])
plt.plot(percentage, label='double Q-learning')



# Q-learning
stats = class_stats()
for k in range(max_k):
    Q1 = Q_estimator()
    Q2 = Q1
    Q = [Q1, Q2]

    for i in range(max_ep):
        state = env.reset()
        while True:
            action = policy.act(state, Q, params)
            next_state, reward, done = env.step(state, action)

            delta = reward+params['gamma']*np.max(Q[0].value[next_state])
            Q[0].value[state][action] += params['alpha']*(delta-Q[0].value[state][action])


            if action == 0 and state == 2:
                stats.nb_left_action[i] += 1
            if state == 2:
                stats.total[i] += 1


            state = next_state
            if done:
                break
# plot results
percentage = np.array([stats.nb_left_action[i]/ stats.total[i] for i in range(max_ep)])
plt.plot(percentage, label = 'Q-learning')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('% left actions from A')
plt.show()
