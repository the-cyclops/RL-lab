import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def on_policy_mc_epsilon_soft( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
    """
    Performs the on policy version of the every-visit MC control starting from the same state
    
    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        eps: random value for the eps-greedy policy (probability of random action)
        gamma: gamma value, the discount factor for the Bellman equation
        
    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """
    #use sample_episode method of gridworld to generate trajectory
    p = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]   
    Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    returns = [[[] for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    
    # Initialize epsilon-soft policy (uniform random policy)
    for s in range(environment.observation_space):
        for a in range(environment.action_space):
            p[s][a] = 1.0 / environment.action_space
    
    for iteration in range(maxiters):
        # Generate episode using current policy
        episode = environment.sample_episode(policy=p)
        
        # Calculate returns for each state-action pair in the episode
        G = 0
        for i in range(len(episode) - 1, -1, -1):  # Work backwards through episode
            state, action, reward = episode[i]
            G = gamma * G + reward
            
            # Update returns and Q-values for every visit
            returns[state][action].append(G)
            Q[state][action] = numpy.mean(returns[state][action])
        
        # Update policy to be epsilon-soft with respect to Q
        for s in range(environment.observation_space):
            # Find best action
            best_action = numpy.argmax(Q[s])
            
            # Set epsilon-soft policy probabilities
            for a in range(environment.action_space):
                if a == best_action:
                    p[s][a] = 1 - eps + eps / environment.action_space
                else:
                    p[s][a] = eps / environment.action_space
    
    deterministic_policy = [numpy.argmax(Q[state]) for state in range(environment.observation_space)]	
    return deterministic_policy
	
	
def on_policy_mc_exploring_starts( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
    """
    Performs the on policy version of the every-visit MC control starting from different states
    
    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        eps: random value for the eps-greedy policy (probability of random action)
        gamma: gamma value, the discount factor for the Bellman equation
        
    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """
    p = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    for i in range(environment.observation_space):
        p[i][0] = 1
    Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    returns = [[[] for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    
    for _ in range(maxiters):
        # Start from a random state with a random action (exploring starts)
        s = environment.random_initial_state()
        environment.robot_state = s
        a = numpy.random.choice(environment.action_space)  # Fixed: use choice instead of random
        
        # Generate episode starting with the random state-action pair
        episode = environment.sample_episode(policy=p, initial_state=s, initial_action=a)
        
        # Calculate returns working backwards through the episode
        G = 0
        for i in range(len(episode) - 1, -1, -1):  # Fixed: iterate backwards through episode length
            state, action, reward = episode[i]
            G = gamma * G + reward  # Fixed: use current reward, not next
            
            # Update returns and Q-values
            returns[state][action].append(G)
            Q[state][action] = numpy.mean(returns[state][action])
            
            # Update policy to be greedy with respect to Q
            best_action = numpy.argmax(Q[state])
            for a_idx in range(environment.action_space):
                p[state][a_idx] = 0  # Reset all probabilities
            p[state][best_action] = 1  # Set greedy action probability to 1
    
    deterministic_policy = [numpy.argmax(Q[state]) for state in range(environment.observation_space)]	
    return deterministic_policy


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the third lesson of the RL-Lab!   *" )
	print( "*            (Monte Carlo RL Methods)            *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n3) MC On-Policy (with exploring starts)" )
	mc_policy = on_policy_mc_exploring_starts( env )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	
	print( "\n3) MC On-Policy (for epsilon-soft policies)" )
	mc_policy = on_policy_mc_epsilon_soft( env )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	

if __name__ == "__main__":
	main()
