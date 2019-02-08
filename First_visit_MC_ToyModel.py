import numpy as np

#generate_episode
def generate_episode():
    s = 0
    episode = []
    r = 1
    prob = 0.75
    while s ==0:
        a = np.random.rand() > prob
        s = a*1
        if s == 1:
            r = 0
        episode.append((s,0,r))
    return episode

max_episode = 1000

N = np.zeros((2,1))
returns_sum = np.zeros((2,1))
Q = np.zeros((2,1))

for i in range(max_episode):
    episode = generate_episode()
    #episode = [(S_t,A_t,R_{t+1}),..., (S_T,A_T,0)]
    states, actions, rewards = zip(*episode)
    #enumerate successive states
    visited_states = []
    for i, state in enumerate(states):
        #first-visit MC
        if state not in visited_states:
            N[state][actions[i]] += 1
            returns_sum[state][actions[i]] += sum(rewards[i:])
            Q[state][actions[i]] = returns_sum[state][actions[i]]/N[state][actions[i]]
        visited_states.append(state)
print(Q)
