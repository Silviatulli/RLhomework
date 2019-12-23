import numpy as np
import matplotlib.pyplot as plt

goal = 100 # define the goal
states = list(range(101)) # create an array of all possible states from 0 to 100
state_values = np.zeros(len(states)) # initialize all state values to 0
rewards= np.zeros(len(states)) # set the reward of all states to 0
rewards[goal] = 1.0 # only rewards the gambler for reaching his goal
head_probability = 0.4  # define the probability of having head
tail_probability = 1 - head_probability # define the probability of having tail

def gambler (goal, states, head_probability, tail_probability):
    epsilon = 1e-8 # define convergence criterion
    overall_error =1 # initialize the overall error
    
    # compute value iteration
    i = 0
    fig1 = plt.figure()
    while overall_error > epsilon: #stop the algorithm when the overall error is smaller than epsilon, so the difference of 2 iterations is less than the defined convergence criterion
        i += 1
        overall_error= 0
        

        
        #For each state find the action values, then update the state values according to the optimal action value (optimal policy)
        for state in states[1:]:
            old_state_value = state_values[state]
            actions = np.arange(min(state, goal-state)+1) # return possible actions given the current state and the goal of the game
            action_values = np.zeros(len(actions)) 
            for action in actions[1:goal]:
                action_values[np.where(actions==action)] = rewards[state]+head_probability*state_values[state+action]+tail_probability*state_values[state-action] #update my action value at the specific position
            if len(action_values[1:]):
                state_values[state] = np.max(action_values[1:])
            else:
                state_values[state] = rewards[state] # if the gambler does not reach the goal gets a reward equal to 0
            overall_error += np.abs(old_state_value - state_values[state])
        
        if i == 1 or i==2 or i ==3:
            plt.plot(state_values, label = 'iteration'+str(i))

    if overall_error < epsilon:
        plt.plot(state_values, label = 'final iteration')
        plt.legend()
        plt.xlabel('state')
        plt.ylabel('state values')
        plt.title('State values per iteration step')
        plt.legend()
        fig1.savefig('ex2'+'.png')
    
    #for each state, get the available actions, the action values and choose the action to maximize action values
    optimal_policy = np.zeros(len(states))
    for state in states[1:-1]:
        actions =np.arange(min(state, goal-state)+1)
        action_values = np.zeros(len(actions))
        for action in actions[1:] :
            action_values[np.where(actions==action)] = rewards[state]+head_probability*state_values[state+action]+tail_probability*state_values[state-action]
        print(action_values)
        optimal_policy[state]=actions[np.argmax(np.round(action_values, 4))] #due to the numerically instability of the gambler problem all the values are rounded up to 4 decimal places	
    
    fig2 = plt.figure()
    plt.plot(optimal_policy)
    plt.legend()
    plt.xlabel('state')
    plt.ylabel('action')
    plt.title('Optimal action per state')
    plt.legend()
    fig2.savefig('optimal_policy'+'.png')

gambler(goal, states, head_probability, tail_probability)
