import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import seaborn as sns
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation):
	"""
	Function that generates a neural network with the given requirements with Keras.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	Returns:
		model: the generated tensorflow model

	"""
	
	# Initialize the neural network
	model = Sequential()
	# ... and crate the input layer ...
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) 
	# ... adding the hidden layers ...
	for _ in range(nLayer):	model.add(Dense(nNodes, activation="relu")) 
	# ... and the output layer
	model.add(Dense(nOutputs, activation=last_activation)) 
	#
	return model


class TorchModel(nn.Module):
	"""
	Class that generates a neural network with PyTorch and specific parameters.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	"""
	
	# Initialize the neural network
	def __init__(self, nInputs, nOutputs, nLayer, nNodes, last_activation):
		
		super(TorchModel, self).__init__()
		self.nLayer = nLayer
		self.last_activation= last_activation

		# input layer
		self.fc1 = nn.Linear(nInputs, nNodes)

		#hidden layers
		for i in range(nLayer):
			layer_name = f"fc{i+2}"
			self.add_module(layer_name, nn.Linear(nNodes, nNodes))  

		#output
		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		for i in range(2, self.nLayer + 2):
			x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
		x = self.output(x)
		#return x if self.last_activation == F.linear else self.last_activation(x, dim=1)
		# Ensure x has at least 2 dimensions before applying softmax
		if self.last_activation == F.softmax:
			if x.dim() == 1:  # Add batch dimension if x is 1D
				x = x.unsqueeze(0)
			return self.last_activation(x, dim=1)
		else:
			return x

# For action selection - use stochastic policy
def select_action(actor_net, state):
	probs = actor_net(state)
	m = Categorical(probs)
	action = m.sample()
	return action.item()

def training_loop(env, actor_net, critic_net, updateRule, frequency=10, episodes=100, keras=True):

	# Reset the global optimizer and memories before the training
	if keras:
		actor_optimizer = tf.keras.optimizers.Adam()
		critic_optimizer = tf.keras.optimizers.Adam()
	else:
		actor_optimizer = optim.Adam(actor_net.parameters()) 
		critic_optimizer = optim.Adam(critic_net.parameters())

	rewards_list, reward_queue = [], collections.deque(maxlen=100)
	memory_buffer = [] # In this exercise the memory buffer contains entries (state, action, reward, next_state, done), not trajectories as in the previous exercise
	for ep in range(episodes):
		
		torch.cuda.empty_cache()
		
		#TODO: reset the environment and obtain the initial state
		ep_reward = 0
		done = False
		state = env.reset()[0]
		#print(type(state)) 
		state = torch.from_numpy(state).to(DEVICE)
		while True:

			#state = torch.from_numpy(state).to(DEVICE)
			#TODO: select the action to perform
			action = select_action(actor_net, state)
			#print( f"action: {action}" )
			#TODO: Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, done, _, _ = env.step(action)
			next_state = torch.from_numpy(next_state).to(DEVICE)
			memory_buffer.append( (state, action, reward, next_state, done) )
			ep_reward += reward

			#TODO: exit condition for the episode
			if done: break

			#TODO: update the current state
			state = next_state


		# Update the reward list to return
		reward_queue.append( ep_reward )
		rewards_list.append( np.mean(reward_queue) )
		if ep % 100 == 0:
			print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )

		# Perform the actual training
		if ep % frequency == 0 and ep != 0: 
			updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, keras)
			memory_buffer = []

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list



def A2C(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, keras, gamma=0.99):

	"""
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
	
	#TODO: implement the update rule for the critic (value function)
	for _ in range(10):
		# Shuffle the memory buffer
		np.random.shuffle( memory_buffer )
		#TODO: extract the information from the buffer
		#memory buffer contains entries (state, action, reward, next_state, done)
		states, actions, rewards, next_states, dones = zip(*memory_buffer)
		#THESE ARE TORCH ONLY, PUT THEM IN THE ELSE BELOW IF YOU WANT TO USE KERAS TOO
		states = torch.stack(states).to(DEVICE)
		actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
		rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
		next_states = torch.stack(next_states).to(DEVICE)
		dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
		
		# Tape for the critic
		# Update the critic
		if keras:
			with tf.GradientTape() as critic_tape:
				#TODO: Compute the target and the MSE between the current prediction
				# and the expected advantage 
				#TODO: Perform the actual gradient-descent process
				raise NotImplementedError
		else:
			# TODO: torch implementation 
			targets = rewards + (1 - dones) * gamma * critic_net(next_states).detach()
			predictions = critic_net(states).squeeze()
			mse = F.mse_loss(predictions, targets)
			critic_optimizer.zero_grad()
			mse.backward()
			critic_optimizer.step()

	#TODO: implement the update rule for the actor (policy function)
	# Update the actor
	if keras:
		with tf.GradientTape() as actor_tape:
			#TODO: compute the log-prob of the current trajectory and 
			# the objective function, notice that:
			# the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
			# multiplied by advantage
			#TODO: compute the final objective to optimize, is the average between all the considered trajectories
			predictions = None 
			probabilities = [entry[actions[idx]] for idx, entry in enumerate(predictions)]
			
			raise NotImplementedError
	else:
		# TODO: torch implementation 
		# For ACTOR UPDATE - Extract data once (no shuffling needed) NOT SURE
		#states, actions, rewards, next_states, dones = zip(*memory_buffer)
		#states = torch.stack(states).to(DEVICE)
		#actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
		#rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
		#next_states = torch.stack(next_states).to(DEVICE)
		#dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
		predictions = actor_net(states)
		#print(f"Predictions: {predictions}")
		probabilities = predictions.gather(-1, actions[:,None]).squeeze()
		log_probabilities = torch.log(probabilities)
		
		with torch.no_grad(): #needed? 
			#adv_a = rewards + gamma * critic_net(next_states)  #.detach().numpy().reshape(-1) 
			#adv_b = critic_net(states)  #.detach().numpy().reshape(-1)

			adv_a = rewards + (1 - dones) * gamma * critic_net(next_states).detach()
			adv_b = critic_net(states).detach()
			advantage = adv_a - adv_b

		objective = -(log_probabilities * (advantage)).mean()
		actor_optimizer.zero_grad()
		objective.backward()
		actor_optimizer.step()
		
	

def main(): 
	print( "\n*************************************************" )
	print( "*  Welcome to the nineth lesson of the RL-Lab!   *" )
	print( "*                    (A2C)                      *" )
	print( "*************************************************\n" )

	print( "*************************************************\n" )
	print("ATTENZIONE HO FATTO SOLO TORCH, COMMENTO TUTTE LE ROBA DI KERAS NEL MAIN")
	print( "*************************************************\n" )

	print(torch.__version__)
	print(torch.cuda.is_available())
	print(f"device: {DEVICE}")
	training_episodes = 5000

	print("\nTraining torch model...\n")
	rewards_torch = []
	for _ in range(3):
		env = gymnasium.make("CartPole-v1")
		actor_net = TorchModel(nInputs=4, nOutputs=2, nLayer=2, nNodes=32, last_activation=F.softmax).to(DEVICE)
		critic_net = TorchModel(nInputs=4, nOutputs=1, nLayer=1, nNodes=32, last_activation=F.linear).to(DEVICE)
		rewards_torch.append(training_loop(env, actor_net, critic_net, A2C, episodes=training_episodes, keras=False))

	#print("\nTraining keras model...\n")
	#rewards_keras = []
	#for _ in range(3):
	#	env = gymnasium.make("CartPole-v1")
	#	actor_net = createDNN( 4, 2, nLayer=2, nNodes=32, last_activation="softmax")
	#	critic_net = createDNN( 4, 1, nLayer=1, nNodes=32, last_activation="linear")
	#	rewards_keras.append(training_loop( env, actor_net, critic_net, A2C, episodes=training_episodes, keras=True))


	# plotting the results
	t = list(range(0, training_episodes))

	data_torch = {'Environment Step': [], 'Mean Reward': []}
	for _, rewards in enumerate(rewards_torch):
		for step, reward in zip(t, rewards):
			data_torch['Environment Step'].append(step)
			data_torch['Mean Reward'].append(reward)
	df_torch = pd.DataFrame(data_torch)

	#data_keras = {'Environment Step': [], 'Mean Reward': []}
	#for _, rewards in enumerate(rewards_keras):
	#	for step, reward in zip(t, rewards):
	#		data_keras['Environment Step'].append(step)
	#		data_keras['Mean Reward'].append(reward)
	#df_keras = pd.DataFrame(data_keras)

	
	# Plotting
	sns.set_style("darkgrid")
	#sns.color_palette("Set2")
	plt.figure(figsize=(8, 6))  # Set the figure size
	sns.lineplot(data=df_torch, x='Environment Step', y='Mean Reward', label='PyTorch', errorbar='se')
	#sns.lineplot(data=df_keras, x='Environment Step', y='Mean Reward', label='Keras', errorbar='se')

	# Add title and labels
	plt.title('Comparison PyTorch-Keras A2C on CartPole-v1')
	plt.xlabel('Episodes')
	plt.ylabel('Mean Reward')

	# Show legend
	plt.legend()

	# Show plot
	plt.show()

if __name__ == "__main__":
	main()	
