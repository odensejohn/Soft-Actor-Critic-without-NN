# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:04:36 2022

@author: odens
Code inspired by https://dubeysarth.github.io/Reinforcement_Learning/Sections/007_Tutorial_Maze.html
"""

# Importing Libraries

from gym import Env
from gym.spaces import Discrete
import numpy as np
import os
import matplotlib.pyplot as plt



#%% Create grid environment
''' 
0 = obstacle
1 = possible path
2 = start position
3 = end position
'''

class GridEnv(Env):
    def __init__(self, step_counter):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(25)
        self.Grid = np.array([
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1]
            ])
        self.mouse = [4,0]
        self.first = [3,4]
        self.second = [1,1]
        self.third = [3,0]
        self.fourth = [4,4]
        self.cheese = [0,4]

        self.state = int(np.arange(25).reshape(5,5)[self.mouse[0],
                                                     self.mouse[1]])
        self.step_counter = step_counter
        self.step_counter_reset = step_counter
        
    def step(self, action):
        # action-> 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:
            new_coor = [self.mouse[0]-1, self.mouse[1]]
        elif action == 1:
            new_coor = [self.mouse[0]+1, self.mouse[1]]
        elif action == 2:
            new_coor = [self.mouse[0], self.mouse[1]-1]
        elif action == 3:
            new_coor = [self.mouse[0], self.mouse[1]+1]
        reward = -1
        if (new_coor[0] != -1) and (new_coor[1] != -1) and (new_coor[0] < 5) and (new_coor[1] < 5):
            if self.Grid[new_coor[0],new_coor[1]] == 1:
                self.mouse = new_coor
                self.state = int(np.arange(25).reshape(5,5)[self.mouse[0], self.mouse[1]])
                reward = -1
            elif new_coor == 0:
                reward = -1
        else:
            reward = -1
            
        if self.mouse == self.first:
            if first_visit[0] == 0:
               first_visit[0] = 1
               reward += -1
            else: 
              reward = -1
              
        if self.mouse == self.second and first_visit[0] == 1:
             if first_visit[1] == 0:
                first_visit[1] = 1
                reward = -1
             else: 
               reward = -1 
        if self.mouse == self.third and first_visit[1] == 1:
             if first_visit[2] == 0:
                first_visit[2] = 1
                reward = -1
             else: 
               reward = -1
        if self.mouse == self.fourth and first_visit[2] == 1:
             if first_visit[3] == 0:
                first_visit[3] = 1
                reward = -1
             else: 
               reward = -1
        if self.mouse == self.cheese and first_visit[3]==1:
            done = True
            reward += 20
        else:
            done = False
        self.step_counter -= 1
        if self.step_counter == 0:
            done = True
            reward = -1
        info = {}
        return self.state, reward, done, info
    
    def render(self, mode = None):
        Grid = np.array(self.Grid)
        print(self.mouse, self.cheese)
        Grid[self.mouse[0],self.mouse[1]] = 2
        Grid[self.cheese[0],self.cheese[1]] = 3
        print(Grid)
        
    def reset(self):
        self.Grid = np.array([
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1]
            ])
        self.mouse = [4,0]
        self.state = int(np.arange(25).reshape(5,5)[self.mouse[0],
                                                     self.mouse[1]])
        self.step_counter = self.step_counter_reset
        return self.state
    
#%% Define actor critic - need to add annotation

class Agent():
    def __init__ (self, lr, gamma, lambda_, n_actions, n_states):
        self.lr = lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_actions = n_actions
        self.n_states = n_states
        self.actor = {}
        self.critic = {}
        self.eligibility_trace = {}
        self.init_actor()
        self.init_critic()
        self.init_eligibility_trace()
        
    # Create an empty state action grid for the actor
    def init_actor(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.actor[(state, action)] = 0.0
    
    # Define a utility grid for the critic
    def init_critic(self):
        for state in range(self.n_states): 
            self.critic[(state)] = 0.0
    
    # Initialise elibility traces to 0
    def init_eligibility_trace(self):
        for state in range(self.n_states):
           for action in range(self.n_actions):
               self.eligibility_trace[(state, action)] = 0       
            
    # Define softmax function
    def softmax(self, x):
        '''
        Compute softmax values of array x.
        @param x the input array
        @return the softmax array
        '''
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    
    # Actor takes the current best action for the state of the grid.  Defined using a softmax approach
             
    def choose_action(self, state):
        actions = np.array([self.actor[(state, a)] for a in range(self.n_actions)])
        action_distribution = self.softmax(actions)
        action = np.random.choice(4, 1, p=action_distribution)
        return action[0]

    
    def learn(self, action, state, reward, state_):
        # Update current state action space in eligibility trace to 1
        self.eligibility_trace[(state, action)] = 1
        # Calculate delta and the update for the function
        delta = self.lr * (reward + self.gamma * self.critic[(state_)] - self.critic[(state)])
        update = self.eligibility_trace[(state, action)] * delta
        # Update the critic and actor
        self.critic[(state)] += update
        self.actor[(state, action)] += update
        
        # decay elibility trace
        self.eligibility_trace = {
            (state, action): self.gamma * self.lambda_ * self.eligibility_trace[(state, action)]
            for (state, action) in self.eligibility_trace
            }
        
#%% Plotting results
def plot_learning_curve(steps_per_episode, scores, name, lr, gamma, lambda_):
    Figure_title = f'Graphs of Score and Steps per Episode\n{name}\n lr:{lr}, gamma={gamma}, lambda={lambda_}'
    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,9))
    #Plotting scores
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax.plot(running_avg, color = "C1")
    ax.set_ylabel('Score', color = "C1")
    ax.tick_params(axis = 'y', colors = "C1")
    ax.grid(axis='x')
    
    #Plotting number of steps per episode
    M = len(steps_per_episode)
    steps_average = np.empty(M)
    for t in range(M):
        steps_average[t] = np.mean(steps_per_episode[max(0, t-100):(t+1)])
    ax2.plot(steps_average, label="C2")
    ax2.set_ylabel("Steps per episode")
    ax2.tick_params(axis = 'y', colors = "C2")
    ax2.grid(axis='x')
    
    fig.align_labels()
    fig.supxlabel("Episodes")
    fig.suptitle(Figure_title, fontsize = 16)
    fig.savefig(name+'.png')
#%% Training code
NAME = 'eligibilitySAC50step50000episode_final20_test2'
if not os.path.exists(NAME):
    os.makedirs(NAME)
env = GridEnv(step_counter=50)
agent = Agent(
    lr = 0.0001, 
    gamma = 0.9, 
    lambda_ = 0.999,
    n_actions = env.action_space.n, 
    n_states = env.observation_space.n
    )
best_score = -np.inf
episodes = 50000

scores, steps_array = [], []
for episode in range(1, episodes+1):
    observation = env.reset()
    done = False
    first_visit=[0,0,0,0]
    score = 0 
    n_steps = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.learn(action, observation, reward, observation_)
        score += reward
        observation = observation_
        n_steps += 1
    scores.append(score)
    steps_array.append(n_steps)

    if episode % 1000 == 0:
        print('Episode:{} Score:{}'.format(episode, score))
        #print(agent.critic)
        #print(agent.actor)
        #trace_dict = {key: round(agent.eligibility_trace[key], 2) for key in agent.eligibility_trace}
        #print(trace_dict)
    if score > best_score:
        best_score = score
#print(scores)
plot_learning_curve(steps_array, scores,  name=(f'{NAME}/{NAME}'),
                    lr=agent.lr, gamma=agent.gamma, lambda_=agent.lambda_)
print('Best Score: {}'.format(best_score))