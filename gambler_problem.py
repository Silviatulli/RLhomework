import numpy as np
import matplotlib.pyplot as plt

goal = 100
states = list(range(101))
head_probability = 0.4
tail_probability = 1 - head_probability

def gambler (goal, states, head_probability, tail_probability):
    state_values = np.zeros(len(states))
    state_values[goal] =1.0
    epsilon =1e-8
    overall_error =1
    
    # compute value iteration
    i = 0
    fig = plt.figure()
    while overall_error > epsilon:
        i += 1
        overall_error= 0
        for state in states:
            old_state_value = state_values[state]
            actions = np.arange(min(state, goal-state)+1)
            action_values = np.zeros(len(actions))#empty array of actions
            for action in actions[1:goal]:
                action_values[np.where(actions==action)] = head_probability*state_values[state+action]+tail_probability*state_values[state-action]#update my action value at the specific position
            if len(action_values[1:]):
                state_values[state] = np.max(action_values[1:])
            overall_error += np.abs(old_state_value - state_values[state])
        if i == 1 or i==2 or i ==3:
            plt.plot(state_values, label = 'iteration'+str(i))
    plt.plot(state_values, label = 'final iteration')
    plt.legend()
    fig.savefig('ex2.png')
    
    #optimal policy
    policy = np.zeros(len(states))
    for state in states[1:-1]:
        actions =np.arange(min(state, goal-state)+1)
        action_values = np.zeros(len(actions))
        for action in actions[1:] :
            action_values[np.where(actions==action)] = head_probability*state_values[state+action]+tail_probability*state_values[state-action]
        print(action_values)
        policy[state]=actions[np.argmax(np.round(action_values, 4))] #the policy will choose the action in relation to the action_values	
    fig1 = plt.figure()
    plt.plot(policy)
    fig1.savefig('policy.png')

gambler(goal, states, head_probability, tail_probability)







