import numpy as np
import gym

def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    grid = [np.linspace(low[i], high[i], bins[i]+1)[1:-1]\
    + offsets[i] for i in np.arange(len(bins))]
    return grid

def create_tilings(low, high, tiling_specs):
    tilings = []
    for specs in tiling_specs:
        bins, offsets = specs
        grid = create_tiling_grid(low, high, bins, offsets)
        tilings.append(grid)
    return tilings

def discretize(sample, grid):
    return tuple(np.digitize(s,g) for s, g in zip(sample,grid))


def tile_encode(sample, tilings, flatten=False):
    state = []
    for grid in tilings:
        state.append(discretize(sample, grid))
        
    if flatten == False:
        return state

    else:
        return [x for coordinates in state for x in coordinates]
#####
### Now use these functions
env = gym.make('MountainCar-v0')
low = env.observation_space.low
high = env.observation_space.high
print(low, high)


n_bins = 8
bins =(n_bins,n_bins)

epsilon = 0.1
gamma = 0.99
alpha = 0.2/3
i_episode = 5000
#manually for now

offset_pos = (env.observation_space.high - env.observation_space.low)/(3*n_bins)
#tiling_specs = [((10,10), (0,0)), ((10,10), (0,0)), ((10,10), (0,0)) ]

tiling_specs = [(bins, -offset_pos),
                (bins, tuple([0.0]*env.observation_space.shape[0])),
                (bins, offset_pos)]


tilings = create_tilings(low, high, tiling_specs)


class Q_table:
    def __init__(self):
        self.Q = [np.zeros((n_bins,n_bins,3)), np.zeros((n_bins,n_bins,3)),np.zeros((n_bins,n_bins,3))]
    def Qvalue(self, state, action):
        states = tile_encode(state, tilings, flatten=False)
        mean = np.mean([table[state+(action,)] for state, table\
           in zip(states,self.Q)])
        return mean
    def Qlearn(self, state, next_state, action, reward):
        states = tile_encode(state, tilings, flatten=False)
        Q_S = [self.Qvalue(next_state, action) for action in np.arange(3)]
        for table, state in zip(self.Q, states):
            table[state+(action,)] += alpha*(reward+gamma*np.max(Q_S)\
                -table[state+(action,)])
    def act(self, state, epsilon):
        rd = np.random.rand()
        if rd < epsilon:
            action = np.random.choice(np.arange(3))
        else:
            states = tile_encode(state, tilings, flatten=False)
            Q_s = [self.Qvalue(state, action) for action in np.arange(3)]
            action = np.random.choice(np.flatnonzero(Q_s == np.max(Q_s)))
        return action


agent = Q_table()


state = env.reset()

average = -200
beta = 0.7
best_score = -np.inf
score = -200
for i in range(i_episode):
    epsilon = epsilon*0.999
    average = beta*score + (1-beta)*average
    if i%100 == 0:
        
        print("\r{}/{}\tcurrent score: {}\tbest: {}\taverage: {}\t min Q: {}"\
            .format(i,i_episode,score,best_score, average, np.min(agent.Q)), end = " ")

    score = 0
    state = env.reset()
    while True:
        action = agent.act(state,epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.Qlearn(state,next_state,action,reward)
        state = next_state
        score += reward
        if done:
            best_score = np.maximum(score, best_score)
            break

print(agent.Q)

while True:
    state = env.reset()
    for t in range(200):
        action = agent.act(state,epsilon)
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break