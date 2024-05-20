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



def createDNN(nInputs, nOutputs, nLayer, nNodes, last_activation):
	"""
	Function that generates a neural network with the given requirements.

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
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) 
	for _ in range(nLayer):	model.add(Dense(nNodes, activation="relu")) 
	model.add(Dense(nOutputs, activation=last_activation)) 
	
	return model

def mse(predicted_value, target):
	"""
	Compute the MSE loss function

	"""
	
	# Compute MSE between the predicted value and the expected labels
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)
	
	# Return the averaged values for computational optimization
	return mse
	

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
		return x if self.last_activation == F.linear else self.last_activation(x, dim=1)


def training_loop(env, actor_net, critic_net, updateRule, frequency=10, episodes=100, keras=True):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the trainign.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	# Get Observation and Action Space
	observation_number = env.observation_space.shape[0]
	action_number = env.action_space.n 

	# Reset the global optimizer and memories before the training
	if keras:
		actor_optimizer = tf.keras.optimizers.Adam()
		critic_optimizer = tf.keras.optimizers.Adam()
	else:
		actor_optimizer = optim.Adam(actor_net.parameters()) 
		critic_optimizer = optim.Adam(critic_net.parameters())

	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
	lenght_list, length_queue = [], collections.deque( maxlen=100 )
	memory_buffer = []
	for ep in range(episodes):

		# Reset the environment and the episode reward before the episode
		state = None #TODO
		ep_reward, ep_length = 0, 0
		while True:

			# Select the action to perform
			try:
				distribution = None
				action = None
			except:
				# to avoid NaN probabilities error
				action = np.random.choice(action_number)


			# Perform the action, store the data in the memory buffer and update the reward
			#TODO
			reward = 0
			ep_reward += reward
			ep_length += 1

			# Exit condition for the episode
			done = None
			if done: break
			state = None

		# Update the reward list to return
		reward_queue.append(ep_reward)
		rewards_list.append(np.mean(reward_queue))
		length_queue.append(ep_length)
		lenght_list.append(np.mean(length_queue))
		print( f"episode {ep:4d}: reward: {ep_reward:5.2f} (averaged: {np.mean(reward_queue):5.2f}), length {ep_length:5.2f} (averaged: {np.mean(length_queue):5.2f})" )

		# Perform the actual training
		if ep % frequency == 0 and ep != 0: 
			updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, keras, observation_number=observation_number)
			memory_buffer = []

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list, lenght_list


def A2C(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, keras, gamma=0.99, observation_number=None):

	"""
	Main update rule for the A2C process. Extract data from the memory buffer and update 
	the newtwork computing the gradient.

	"""

	# Iterate 10 times the update for the value function
	for _ in range(10):

		# Shuffle the memory buffer and extract the information from the buffer
		#TODO

		if keras:
			# Tape for the critic
			#TODO
			pass
		else:
			#TODO
			pass


	# Extract the information from the buffer for the policy update
	#TODO
	

	if keras: 
		# Tape for the actor
		#TODO
		pass
		
	else:
		#TODO
		pass
	

class OverrideReward( gymnasium.wrappers.NormalizeReward ):


	def step(self, action):
		previous_observation = np.array(self.env.state, dtype=np.float32)
		observation, reward, terminated, truncated, info = self.env.step(action)
	
		# Extract the information about the state from the obeservation
		position, velocity = observation[0], observation[1]

		#TODO override reward function here...

		return observation, reward, terminated, truncated, info
	

def main(): 
	print( "\n***************************************************" )
	print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
	print( "*                 (DRL in Practice)               *" )
	print( "***************************************************\n" )

	training_episodes = 2500

	# Crete the environment adding the wrapper for the custom reward function
	gymnasium.envs.register(
		id='MountainCarMyVersion-v0',
		entry_point='gymnasium.envs.classic_control:MountainCarEnv',
		max_episode_steps=1000
	)

	
	# Create the networks and perform the actual training
	print("\nTraining torch model...\n")
	ep_lengths_torch = []
	for run in range(3):
		env = gymnasium.make("MountainCarMyVersion-v0")
		env = OverrideReward(env)
		actor_net = TorchModel(nInputs=env.observation_space.shape[0], nOutputs=env.action_space.n, nLayer=2, nNodes=32, last_activation=F.softmax)
		critic_net = TorchModel(nInputs=env.observation_space.shape[0], nOutputs=1, nLayer=1, nNodes=32, last_activation=F.linear)
		_, ep_lengths = training_loop(env, actor_net, critic_net, A2C, frequency=5, episodes=training_episodes, keras=False)
		ep_lengths_torch.append(ep_lengths)

	# 	# Save the trained neural network
	# 	torch.save(actor_net, f"MountainCarActor_{run}.pth")

	print("\nTraining keras model...\n")
	ep_lengths_keras = []
	for run in range(3):
		env = gymnasium.make("MountainCarMyVersion-v0")
		env = OverrideReward(env)
		actor_net = createDNN(nInputs=env.observation_space.shape[0], nOutputs=env.action_space.n, nLayer=1, nNodes=32, last_activation="softmax")
		critic_net = createDNN(nInputs=env.observation_space.shape[0], nOutputs=1, nLayer=1, nNodes=32, last_activation="linear")
		_, ep_lengths = training_loop(env, actor_net, critic_net, A2C, frequency=5, episodes=training_episodes)
		ep_lengths_keras.append(ep_lengths)

		# Save the trained neural network
		#actor_net.save(f"MountainCarActor_{run}.h5")

	# plotting the results
	t = list(range(0, training_episodes))

	data_torch = {'Episode': [], 'Mean ep_length': []}
	for _, mean_ep_lengths in enumerate(ep_lengths_torch):
		for episode, ep_length in zip(t, mean_ep_lengths):
			data_torch['Episode'].append(episode)
			data_torch['Mean ep_length'].append(ep_length)
	df_torch = pd.DataFrame(data_torch)

	data_keras = {'Episode': [], 'Mean ep_length': []}
	for _, mean_ep_lengths in enumerate(ep_lengths_keras):
		for episode, ep_length in zip(t, mean_ep_lengths):
			data_keras['Episode'].append(episode)
			data_keras['Mean ep_length'].append(ep_length)
	df_keras = pd.DataFrame(data_keras)

	# Plotting
	sns.set_style("darkgrid")
	#sns.color_palette("Set2")
	plt.figure(figsize=(8, 6))  # Set the figure size
	sns.lineplot(data=df_torch, x='Episode', y='Mean ep_length', label='PyTorch', errorbar='se')
	sns.lineplot(data=df_keras, x='Episode', y='Mean ep_length', label='Keras', errorbar='se')

	# Add title and labels
	plt.title('Comparison PyTorch-Keras A2C on MountainCar-v0')
	plt.xlabel('Episodes')
	plt.ylabel('Mean episode length')

	# Show legend
	plt.legend()

	# Show plot
	plt.show()
	


if __name__ == "__main__":
	main()	
