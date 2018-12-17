import numpy as np
import matplotlib
import matplotlib.pyplot as plt

goal = 100
states = list(range(101))
head_probability = 0.4
tail_probability = 1 - head_probability

def gambler ():
	state_values = np.zeros(states) #initialize at 0 because we don't know how much good it is to stay in a certain state
	state_values[goal] = 1.0 #reward for achiving the goal
	epsilon = 1e-8 #final value for the iteration
	overall_error = 1 #initial value for the iteration


	#value iteration
	i = 0
	fig = plt.figure()
	while overall_error > epsilon:
		i += 1
		overall_error = 0
		for state in states:
			old_state_value = state_values[state]
			actions = np.arange(min(state, goal-state)+1)#for example if i'm in state 90, I can have ..[0,..., 10]
			action_values = np.zeros(len(actions)) #empty array of actions
			for action in actions[1:goal]:
				action_values[np.where(actions==action)] = head_probability*state_values[state+action]+tail_probability*state_values[state-action] #update my action value at the specific position
			state_values[state] = np.max(actions_values)
			overall_error += np.abs(old_state_value - state_values[state])
		if i == 1 or i == 2 or i == 3:
			plt.plot(state_values)
	plt.plot(state_values)
	fig.savefig('state_values.png')



	#optimal policy (the policy gives you only an action)
	policy = np.zeros(states)
	for state in states:
		actions = np.arange(min(state, goal-state)+1)#for example if i'm in state 90, I can have ..[0,..., 10]
		action_values = np.zeros(len(actions)) #empty array of actions
		for action in actions[1:]:
			action_values[np.where(actions==action)] = head_probability*state_values[state+action]+tail_probability*state_values[state-action]
		policy[state]=np.where(action_values==np.max(action_values))#the policy will choose the action in relation to the action_values	
	fig1 = plt.figure()
	plt.plot(policy)#policy will print the best actions
	fig1.savefig('policy.png')










