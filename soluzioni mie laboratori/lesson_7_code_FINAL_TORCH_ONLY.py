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
import seaborn as sns
import pandas as pd

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cuda is unefficient for this task, so we use CPU
# it is une
device = "cpu"
def createDNN_keras(nInputs, nOutputs, nLayer, nNodes):
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
	#
	# YOUR CODE HERE!
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
	def __init__(self, nInputs, nOutputs, nLayer, nNodes):
		super(TorchModel, self).__init__()
		self.fc1 = nn.Linear(nInputs, nNodes)
		#
		# YOUR CODE HERE!
		#
		self.relu1 = nn.ReLU()
		self.fch = nn.ModuleList()
		for i in range(nLayer):
			self.fch.append(nn.Linear(nNodes, nNodes))
			self.fch.append(nn.ReLU())
			
		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		#
		# YOUR CODE HERE!
		#
		x = self.fc1(x)
		x = self.relu1(x)
		# Loop over the hidden layers
		for layer in self.fch:
			x = layer(x)
		return self.output(x)	


def mse(network, dataset_input, target, keras=True):
	"""
	Compute the MSE loss function

	"""
	
	# Compute the predicted value, over time this value should
	# looks more like to the expected output (i.e., target)
	predicted_value = network(dataset_input)
	if keras:
		# Compute MSE between the predicted value and the expected labels
		mse = tf.math.square(predicted_value - target)
		mse = tf.math.reduce_mean(mse)
	else:
		# Compute MSE between the predicted value and the expected labels
		mse = nn.MSELoss(reduction='mean')(predicted_value, target)
		#mse = torch.mean(mse) # This is not needed in PyTorch, as the loss function already returns the mean
	# Return the averaged values for computational optimization
	return mse


def training_loop(env, neural_net, updateRule, keras=True, eps=0.99, updates=1, episodes=100):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	#TODO: initialize the optimizer 
	if keras:
		optimizer = None
	else:
		optimizer = optim.Adam(neural_net.parameters(), lr=0.003)

	eps_decay = 0.97 # decay to have similar patter to professor's code
	rewards_list, memory_buffer = [], collections.deque( maxlen=1000 )
	averaged_rewards = []
	for ep in range(episodes):

		torch.cuda.empty_cache()	
		#TODO: reset the environment and obtain the initial state
		state = env.reset()[0] 
		ep_reward = 0
		while True:
			env
			#TODO: select the action to perform exploiting an epsilon-greedy strategy
			if np.random.rand() < eps : action = env.action_space.sample()
			else:
				if keras: pass
				else:
					state_tensor= torch.tensor(state).to(device)
					action = neural_net(state_tensor).argmax().item() 

			#TODO: update epsilon value
			#eps *= eps_decay

			#TODO: Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, done, truncated, _ = env.step(action)
			done = done or truncated
			memory_buffer.append( (state, action, next_state, reward, done) )
			ep_reward += reward

			# Perform the actual training
			for _ in range(updates):
				#TODO: call the update rule...
				updateRule(neural_net, keras, memory_buffer, optimizer)
				

			#TODO: modify the exit condition for the episode
			if done: break

			#TODO: update the current state
			state = next_state

		# Update the reward list to return
		rewards_list.append(ep_reward)
		averaged_rewards.append(np.mean(rewards_list))
		print( f"episode {ep:2d}: mean reward: {averaged_rewards[-1]:3.2f}, eps: {eps:3.2f}" )

		#TODO: update epsilon value
		eps *= eps_decay

	# Close the enviornment and return the rewards list
	env.close()
	return averaged_rewards


def DQNupdate(neural_net, keras, memory_buffer, optimizer, batch_size=64, gamma=0.99):

	"""
	Main update rule for the DQN process. Extract data from the memory buffer and update 
	the newtwork computing the gradient.

	"""

	if len(memory_buffer) < batch_size: return

	indices = np.random.randint( len(memory_buffer), size=batch_size)
	for idx in indices: 

		#TODO: extract data from the buffer 
		#state, action, reward, next_state, done = None, None, None, None, None
		#i used this convenction above, i stick to it
		state, action, next_state, reward, done = memory_buffer[idx]
		
		#TODO: compute the target for the training
		if keras:
			target = None
		else:
			state_tensor = torch.tensor(state).to(device)
			next_state_tensor = torch.tensor(next_state).to(device)
			target = neural_net(state_tensor)

		
		#TODO: update target using the update rule...
		if done:
			target[action] = reward
		else:
			if keras:
				max_q = neural_net(next_state).max().item()
			else: 
				max_q = neural_net(next_state_tensor).max().item()
			target[action] = reward + gamma * max_q

		#TODO: compute the gradient and perform the backpropagation step using the selected framework
		if keras:
			with tf.GradientTape() as tape:
				objective = mse(neural_net, state, target, keras)

		else:
			mse_loss = mse(neural_net, state_tensor, target, keras=False)
			optimizer.zero_grad()
			mse_loss.backward()
			optimizer.step()
			


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the seventh lesson of the RL-Lab!   *" )
	print( "*               (Deep Q-Network)                 *" )
	print( "**************************************************\n" )

	print( "*************************************************\n" )
	print("ATTENZIONE HO FATTO SOLO TORCH, COMMENTO TUTTE LE ROBA DI KERAS NEL MAIN")
	print( "*************************************************\n" )

	training_steps = 50
	
	# setting DNN configuration
	nInputs=4
	nOutputs=2
	nLayer=2
	nNodes=32 

	print(torch.__version__)
	print(torch.cuda.is_available())
	print(f"device: {device}")

	print("\nTraining torch model...\n")
	rewards_torch = []
	dummy_net = TorchModel(nInputs, nOutputs, nLayer, nNodes)
	print(dummy_net)
	hidden_layers_count = len(dummy_net.fch)
	hidden_layers_count = len([layer for layer in dummy_net.fch if isinstance(layer, nn.Linear)])
	print(f"\tNumber of hidden layers: {hidden_layers_count}")
	TOT_ITERATIONS = 10
	for _ in range(TOT_ITERATIONS):
		print( "*************************************************\n" )
		print(f"iteration {_+1} of {TOT_ITERATIONS}, total episodes for each iteration: {training_steps}")
		print( "*************************************************\n" )
		env = gymnasium.make("CartPole-v1")#, render_mode="human" )
		neural_net_torch = TorchModel(nInputs, nOutputs, nLayer, nNodes).to(device)
		rewards_torch.append(training_loop(env, neural_net_torch, DQNupdate, keras=False, episodes=training_steps))

	#print("\nTraining keras model...\n")
	#rewards_keras = []
	#for _ in range(TOT_ITERATIONS):
	#	env = gymnasium.make("CartPole-v1")#, render_mode="human" )
	#	neural_net_keras = createDNN_keras(nInputs, nOutputs, nLayer, nNodes)
	#	rewards_keras.append(training_loop(env, neural_net_keras, DQNupdate, keras=True, episodes=training_steps))


	# plotting the results
	t = list(range(0, training_steps))

	data = {'Environment Step': [], 'Mean Reward': []}
	for _, rewards in enumerate(rewards_torch):
		for step, reward in zip(t, rewards):
			data['Environment Step'].append(step)
			data['Mean Reward'].append(reward)
	df_torch = pd.DataFrame(data)

	#data_keras = {'Environment Step': [], 'Mean Reward': []}
	#for _, rewards in enumerate(rewards_keras):
	#	for step, reward in zip(t, rewards):
	#		data_keras['Environment Step'].append(step)
	#		data_keras['Mean Reward'].append(reward)
	#df_keras = pd.DataFrame(data_keras)

	# Plotting
	sns.set_style("darkgrid")
	plt.figure(figsize=(8, 6))  # Set the figure size
	sns.lineplot(data=df_torch, x='Environment Step', y='Mean Reward', label='torch', errorbar='se')
	#sns.lineplot(data=df_keras, x='Environment Step', y='Mean Reward', label='keras', errorbar='se')

	# Add title and labels
	plt.title('Comparison Keras and PyTorch on CartPole-v1')
	plt.xlabel('Episodes')
	plt.ylabel('Mean Reward')

	# Show legend
	plt.legend()

	# Show plot
	plt.show()
	plt.savefig('comparison.pdf')


if __name__ == "__main__":
	main()	
