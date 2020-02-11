import random
import numpy as np
import matplotlib.pyplot as plt

cols  = 12
rows = 4
start = (rows-1,0)
goal = (rows-1, cols-1)



class Grid:
    # Initialize the grid
    def __init__(self):
        self.cols = cols
        self.rows = rows

    # Define the starting point for each iteration
    def reset(self):
        self.X = 0
        self.Y = 0
        self.state = np.matrix([self.X, self.Y])
        return self.state

    # Define the agent's action
    def step(self, action):
        self.ACTIONS = ["Up", "Down", "Left", "Right"]
        action = self.ACTIONS[action]
        if action == "Up":
            if self.Y < (self.rows - 1):
                self.Y += 1
        elif action == "Down":
            if self.Y > 0:
                self.Y -= 1
        elif action == "Right":
            if self.X < (self.cols - 1):
                self.X += 1
        elif action == "Left":
            if self.X > 0:
                self.X -= 1

        # Value of reward in each part of the grid
        if self._inside_cliff(self.X, self.Y):
            reward = -100
            self.X = 0
            self.Y = 0
        else:
            reward = -1
        self.state = np.matrix([self.X, self.Y])

        # The following statement define the goal
        if (self.X == (self.cols - 1)) and (self.Y == 0):
            done = 1
        else:
            done = 0

        return self.state, reward, done, None

    # The following function define the area of the cliff
    def _inside_cliff(self, X, Y):
        if (Y == 0) and (X > 0) and (X < (self.cols - 1)):
            return True
        else:
            return False

class Agent:
    def __init__(self, agent_type = "SARSA"):
        self.agent_type = agent_type
        self._build_model()
        self.learningrate = 0.5 # learning rate
        self.discount = 1 # discount factor
        self.epsilon = 0.15 #epsilon greedy

    def _build_model(self):
        # Initialize the q values all with zeros
        self.qvalues = np.zeros([cols, rows, 4])

    def _choose_action(self, state):
        if np.random.rand() <= self.epsilon: #epsilon 0.15 policy
            action = random.randrange(4)
        else:
            action = np.argmax(self.predict(state))  #epsilon greedy policy
        return action

    def predict(self, state):
        ret_val = self.qvalues[state[0, 0], state[0, 1], :]
        return ret_val

    def init_episode(self, env):
        if self.agent_type == "SARSA":
            return self._init_episode_sarsa_qlearning(env)
        if self.agent_type == "Q-Learning":
            return self._init_episode_sarsa_qlearning(env)

    def _init_episode_sarsa_qlearning(self, env):
        self.state = env.reset()
        self.action = self._choose_action(self.state)

    def train_step(self, env):
        if self.agent_type == "SARSA":
            return self._train_step_sarsa(env)
        if self.agent_type == "Q-Learning":
            return self._train_step_qlearning(env)

    def _train_step_sarsa(self, env):
        new_state, reward, done, _ = env.step(self.action)
        new_action = self._choose_action(new_state)
        # Q(S;A)<-Q(S;A) + alfa[R + ganma*Q(S';A') - Q(S;A)]
        self.qvalues[self.state[0,0], self.state[0,1], self.action] \
            += self.learningrate* \
            (reward + self.discount*self.predict(new_state)[new_action] \
            - self.predict(self.state)[self.action])
        self.state = new_state
        self.action = new_action
        return new_state, reward, done, self.action, self.epsilon

    def _train_step_qlearning(self, env):
        self.action = self._choose_action(self.state)
        new_state, reward, done, _ = env.step(self.action)
        # Q(S;A)<-Q(S;A) + alfa[R + ganma*maxQ(S';a) - Q(S;A)]
        self.qvalues[self.state[0,0], self.state[0,1], self.action] \
            += self.learningrate* \
            (reward + self.discount*np.amax(self.predict(new_state)) \
            - self.predict(self.state)[self.action])
        self.state = new_state
        return new_state, reward, done, self.action, self.epsilon


if __name__ == "__main__":
    agent_types = ["SARSA","Q-Learning"]
    num_runs = 10000
    num_episodes = 500
    episode_reward_average = {}
    episode_reward_average[0] = np.zeros([num_episodes])#average reward for SARSA
    episode_reward_average[1] = np.zeros([num_episodes])#average reward for Q-Learning

    
    # Train
    for j in range(num_runs):
        print("Run #" + str(j))
        episode_reward = {}
        for i in range(len(agent_types)):
            env = Grid()
            agent = Agent(agent_types[i])
            episode_reward[i] = np.zeros([num_episodes])
            for e in range(num_episodes):
                state = agent.init_episode(env)
                # Here the plot can be added for the initial state
                done = False
                while not done:
                    state, reward, done, action, epsilon = agent.train_step(env)
                    episode_reward[i][e] += reward
            if i == 0:
                episode_reward_average[0] = np.add(episode_reward_average[0], episode_reward[i])
                print(action)
            else:
                episode_reward_average[1] = np.add(episode_reward_average[1], episode_reward[i])
                print(action)

    # Get the average
    episode_reward_average[0] = np.true_divide(episode_reward_average[0], num_runs)
    episode_reward_average[1] = np.true_divide(episode_reward_average[1], num_runs)
    print("Total reward for SARSA:", episode_reward_average[0],"N:", num_runs)
    print("Total reward for Q-learning:", episode_reward_average[1],"N:", num_runs)

    # Plot Rewards
    #fig, ax = plt.subplots()
    #fig.suptitle('Rewards')
    plt.plot(episode_reward_average[1], color='blue', label='qlearning')
    plt.plot(episode_reward_average[0], color='orange', label='sarsa')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.ylim(-125, 0)
    plt.legend() 
    plt.show()
