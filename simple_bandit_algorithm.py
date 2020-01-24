import numpy as np
import matplotlib.pyplot as plt

class bandit():
    def __init__(self, k):
        self.means = np.random.standard_normal(size=k)
        self.best_action = np.argmax(self.means)
        self.counter = 0
        self.optimal_actions = []

    def rewards(self, action):
        if action == self.best_action:
            self.counter += 1
        self.optimal_actions.append(self.counter)

        return np.random.normal(loc=self.means[action], scale=1.0)

class agent():
    def __init__(self, epsilon, k):
        self.epsilon = epsilon
        self.k = k
        self.Q = np.zeros(k) #only one state
        self.N = np.zeros(k)

    def act(self):
        action = np.random.choice([
            np.random.choice(np.flatnonzero(self.Q==np.max(self.Q))),
            np.random.randint(self.k)
            ],
            p=[1-self.epsilon, self.epsilon])
        return action
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (1/self.N[action])*(reward - self.Q[action])


data = np.zeros(1000)
i_max = 2000
for i in np.arange(i_max):
    local = []
    #reset agent and env
    my_agent = agent(epsilon=0.1, k=10)
    env = bandit(k=10)
    for t in np.arange(1000):
        action = my_agent.act()
        reward = env.rewards(action)
        my_agent.update(action, reward)
    local = [env.optimal_actions[i]/(i+1) for i in np.arange(len(env.optimal_actions))]
    data += np.array(local)

plt.plot(100*np.array(data)/i_max)
plt.ylabel('$\%$ optimal action')
plt.xlabel('steps')
plt.ylim([0,100])
plt.show()