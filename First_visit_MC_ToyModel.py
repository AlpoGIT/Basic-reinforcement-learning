import numpy as np
from collections import defaultdict

#general initialization
Q = defaultdict(lambda : np.zeros(2))
N = defaultdict(lambda : np.zeros(2))
returns_sum = defaultdict(lambda : np.zeros(2))

#Generate an episode (unknown to the agent)
def generate_episode():
    episode = []
    state = 0
    while state == 0:
        episode.append((state,0,1))
        state = np.random.choice([0,1],p=[0.75,0.25])
    episode.append((1,0,0))
    return episode

#First-visit MC algorithm
for i in range(20000):
    episode = generate_episode()
    states, actions, rewards = zip(*episode)
    visited_states = []
    for i, state in enumerate(states):
        if state not in visited_states:
            visited_states.append(state)
            N[state][actions[i]] += 1
            returns_sum[state][actions[i]] += np.sum(rewards[i:])
            Q[state][actions[i]] = returns_sum[state][actions[i]]/N[state][actions[i]]

#Print the value of Q(s,a) for s = 0 and a = 0
#i.e. the expected number of steps before reaching the final states. It should be 4.
print(Q[0][0])
