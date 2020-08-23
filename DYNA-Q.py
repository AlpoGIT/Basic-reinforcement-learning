import gym
from collections import defaultdict
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')

alpha = 0.1
gamma = 0.995

def agent_action(state, eps, Q):
    if np.random.uniform(0,1) > eps:
        a = np.flatnonzero(Q[state]==np.max(Q[state]))
        return np.random.choice(a)
    else:
        return np.random.choice([0,1,2,3])

def loop(dyna_L):
    Q = defaultdict(lambda : np.zeros(4))
    action_taken = defaultdict(lambda : [])
    mean_score = deque(maxlen=100)

    eps = 0.2
    model = defaultdict(lambda : (0., 0)) # (r, s') = model[(s,a)]
    observed = []

    for i in range(2000):
        eps = eps*0.95
        state = env.reset()
        total_reward = 0
        while True:
            action = agent_action(state, eps, Q)
            next_state, reward, done, info = env.step(action)

            # simplify
            model[(state, action)] = (reward, next_state)
            if state not in observed:
                observed.append(state)
            if action not in action_taken[state]:
                action_taken[state].append(action)

            # learning
            Q[state][action] += alpha*(reward + gamma*np.max(Q[next_state]) - Q[state][action])

            total_reward += reward
            state = next_state

            # planning
            for _ in range(dyna_L):
                s = np.random.choice(observed)
                a = np.random.choice(action_taken[s])
                r, next_s = model[(s,a)]
                Q[s][a] += alpha*(r + gamma*np.max(Q[next_s]) - Q[s][a])
            
            if done:
                mean_score.append(total_reward)
                #if i%200 == 0:
                #    print('{}'.format(np.mean(mean_score)), end='\n')
                break

        if np.mean(mean_score) >= -13.0:
            print('solved in {} episodes'.format(i))
            return mean_score, Q, i

    return mean_score, Q, i


mean_duration = []
error = []
duration = []
dyna_Ls = np.arange(0,20,1)
for dyna_L in dyna_Ls:
    print("Dyna length:\t",dyna_L)
    
    duration = []
    for t in range(50):
        #print('sample no:\t',t+1)
        _, Q, i = loop(dyna_L)
        duration.append(i)
    mean_duration.append(np.mean(duration))
    error.append(np.std(duration))
    print('mean duration:\t', np.mean(duration))
mean_duration = np.array(mean_duration)
error = np.array(error)
plt.plot(dyna_Ls,mean_duration)
plt.fill_between(dyna_Ls, mean_duration-error, mean_duration+error, alpha=0.5)
plt.xlabel('Dyna Length')
plt.ylabel('Episode mean duration')
plt.savefig('Optimal_Dyna_Length.png')
plt.show()

#state = env.reset()
#eps = 0.
#while True:
#    action = agent_action(state, eps, Q)
#    next_state, reward, done, info = env.step(action)
#    state = next_state
#    env.render()
#    if done:
#        break
