import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld

import numpy as np
import matplotlib.pyplot as plt


def plot_cumulative_rewards(cumulative_rewards_dyna_q, cumulative_rewards_dyna_q_plus):
    """
    Plots cumulative rewards over time steps.

    Args:
        cumulative_rewards_dyna_q: list of Dyna-Q rewards.
        cumulative_rewards_dyna_q_plus: list of Dyna-Q+ rewards.
    """

    time_steps_dyna_q = np.arange(len(cumulative_rewards_dyna_q))
    time_steps_dyna_q_plus = np.arange(len(cumulative_rewards_dyna_q_plus))

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps_dyna_q, cumulative_rewards_dyna_q, marker='o', linestyle='-', color='b', label='Dyna-Q')
    plt.plot(time_steps_dyna_q_plus, cumulative_rewards_dyna_q_plus, marker='x', linestyle='--', color='r', label='Dyna-Q+')
    plt.title('Cumulative Rewards Over Time Steps', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Cumulative Rewards', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()


def epsilon_greedy(q, state, epsilon):
	"""
	Epsilon-greedy action selection function
	
	Args:
		q: q table
		state: agent's current state
		epsilon: epsilon parameter
	
	Returns:
		action id
	"""
	if numpy.random.random() < epsilon:
		return numpy.random.choice(q.shape[1])
	return q[state].argmax()


def dynaQ(environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99):
    """
    Implements the DynaQ algorithm
    
    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        n: steps for the planning phase
        eps: random value for the eps-greedy policy (probability of random action)
        alfa: step size for the Q-Table update
        gamma: gamma value, the discount factor for the Bellman equation
        
    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
        cumulative_rewards: list of cumulative rewards collected every 25 steps
    """	

    Q = numpy.zeros((environment.observation_space, environment.action_space))
    M = {}  # keys are (state,action) while values are (next_state,reward)
    
    cumulative_rewards = []
    episode_reward = 0
    evaluation_interval = 25  # Collect rewards every 25 steps
    
    for iteration in range(maxiters):
        state = environment.robot_state
        action = epsilon_greedy(Q, state, eps)
        next_state = environment.sample(action, state)
        reward = environment.R[next_state]
        
        # Accumulate reward for this step
        episode_reward += reward
        
        # Handle terminal states correctly
        if environment.is_terminal(next_state):
            # For terminal states, there's no future value (Q_next = 0)
            Q[state, action] += alfa * (reward - Q[state, action])
            environment.robot_state = environment.start_state
            
            # Episode ended, record cumulative reward if it's time
            if (iteration + 1) % evaluation_interval == 0:
                cumulative_rewards.append(episode_reward)
            episode_reward = 0  # Reset for next episode
        else:
            Q[state, action] += alfa * (reward + gamma * Q[next_state].max() - Q[state, action])
            M[(state, action)] = (next_state, reward)
            environment.robot_state = next_state
            
            # Record cumulative reward at intervals (during episodes)
            if (iteration + 1) % evaluation_interval == 0:
                cumulative_rewards.append(episode_reward)
        
        # Planning phase - only if model has data
        if M:  # Check if model is not empty
            for _ in range(n):
                model_keys = tuple(M.keys())
                random_idx = numpy.random.randint(len(model_keys))
                state_action = model_keys[random_idx]
                state, action = state_action
                next_state, reward = M[state_action]
                
                # Update Q-value using model
                if environment.is_terminal(next_state):
                    Q[state, action] += alfa * (reward - Q[state, action])
                else:
                    Q[state, action] += alfa * (reward + gamma * Q[next_state].max() - Q[state, action])
    
    policy = Q.argmax(axis=1)
    return policy, cumulative_rewards


def dynaQplus(environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99, k=0.01):
    """
    Implements the DynaQ+ algorithm with exploration bonus
    
    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        n: steps for the planning phase
        eps: random value for the eps-greedy policy (probability of random action)
        alfa: step size for the Q-Table update
        gamma: gamma value, the discount factor for the Bellman equation
        k: exploration bonus coefficient
        
    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
        cumulative_rewards: list of cumulative rewards collected every 25 steps
    """	

    Q = numpy.zeros((environment.observation_space, environment.action_space))
    M = {}  # keys are (state,action) while values are (next_state,reward)
    last_visited = {}  # keys are (state,action), values are last iteration when visited
    
    cumulative_rewards = []
    episode_reward = 0
    evaluation_interval = 25  # Collect rewards every 25 steps
    
    for iteration in range(maxiters):
        state = environment.robot_state
        action = epsilon_greedy(Q, state, eps)
        next_state = environment.sample(action, state)
        reward = environment.R[next_state]
        
        # Accumulate reward for this step
        episode_reward += reward
        
        # Update last visited time for this state-action pair
        last_visited[(state, action)] = iteration
        
        # Handle terminal states correctly
        if environment.is_terminal(next_state):
            # For terminal states, there's no future value (Q_next = 0)
            Q[state, action] += alfa * (reward - Q[state, action])
            environment.robot_state = environment.start_state
            
            # Episode ended, record cumulative reward if it's time
            if (iteration + 1) % evaluation_interval == 0:
                cumulative_rewards.append(episode_reward)
            episode_reward = 0  # Reset for next episode
        else:
            Q[state, action] += alfa * (reward + gamma * Q[next_state].max() - Q[state, action])
            M[(state, action)] = (next_state, reward)
            environment.robot_state = next_state
            
            # Record cumulative reward at intervals (during episodes)
            if (iteration + 1) % evaluation_interval == 0:
                cumulative_rewards.append(episode_reward)
        
        # Planning phase - only if model has data
        if M:  # Check if model is not empty
            for _ in range(n):
                model_keys = tuple(M.keys())
                random_idx = numpy.random.randint(len(model_keys))
                state_action = model_keys[random_idx]
                state, action = state_action
                next_state, original_reward = M[state_action]
                
                # Calculate exploration bonus based on time since last visit
                tau = iteration - last_visited.get(state_action, 0)
                exploration_bonus = k * numpy.sqrt(tau)
                augmented_reward = original_reward + exploration_bonus
                
                # Update Q-value using model with exploration bonus
                if environment.is_terminal(next_state):
                    Q[state, action] += alfa * (augmented_reward - Q[state, action])
                else:
                    Q[state, action] += alfa * (augmented_reward + gamma * Q[next_state].max() - Q[state, action])
    
    policy = Q.argmax(axis=1)
    return policy, cumulative_rewards


def main():
	print( "\n************************************************" )
	print( "*   Welcome to the fifth lesson of the RL-Lab!   *" )
	print( "*                  (Dyna-Q)                      *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld( deterministic=True )
	env.render()

	print( "\n5) Dyna-Q" )
	dq_policy_n00, _ = dynaQ( env, n=0  )
	dq_policy_n25, _ = dynaQ( env, n=25  )
	dq_policy_n50, dq_rewards = dynaQ( env, n=50  )
	env.render_policy( dq_policy_n50 )
	
	print( "\n5) Dyna-Q+" )
	dqp_policy_n00, _ = dynaQplus( env, n=0 )
	dqp_policy_n25, _ = dynaQplus( env, n=25 )
	dqp_policy_n50, dqp_rewards = dynaQplus( env, n=50 )
	env.render_policy( dqp_policy_n50 )
	print()
	
	print( f"\tExpected Dyna-Q reward with n=0:", env.evaluate_policy(dq_policy_n00) )
	print( f"\tExpected Dyna-Q reward with n=25:", env.evaluate_policy(dq_policy_n25) )
	print( f"\tExpected Dyna-Q reward with n=50:", env.evaluate_policy(dq_policy_n50) )
	
	print()
	
	print( f"\tExpected Dyna-Q+ reward with n=0:", env.evaluate_policy(dqp_policy_n00) )
	print( f"\tExpected Dyna-Q+ reward with n=25:", env.evaluate_policy(dqp_policy_n25) )
	print( f"\tExpected Dyna-Q+ reward with n=50:", env.evaluate_policy(dqp_policy_n50) )
	
	plot_cumulative_rewards(dq_rewards, dqp_rewards)
	

if __name__ == "__main__":
	main()
