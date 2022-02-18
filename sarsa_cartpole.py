import numpy as np 
import gym 
import math 
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
#print(env.action_space.n)

LEARNING_RATE = 0.1

DISCOUNT = 1
EPISODES = 50000
CHECKPOINT_INTERVAL=1000
total_reward = 0
prior_reward = 0

Observation = [50, 50, 100, 100]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

epsilon = 0.5
epsilon_decay_value=0.9

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = state/np_array_win_size 
    return tuple(discrete_state.astype(np.int))

reward_record=[]
mean_reward_record=[]

for episode in range(EPISODES + 1): #go through the episodes
    discrete_state = get_discrete_state(env.reset()) #get the discrete start for the restarted environment 
    done = False
    episode_reward = 0 #reward starts as 0 for each episode

    if episode % CHECKPOINT_INTERVAL == 0: 
        print("Episode: " + str(episode))

    while not done: 

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) 
        else:
            action = np.random.randint(0, env.action_space.n) #a random action

        new_state, reward, done, _ = env.step(action) #step action to get new states, reward, and the "done" status.

        episode_reward += reward #add the reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % CHECKPOINT_INTERVAL == 0: #render
            env.render()

        if not done: #update q-table
            future_q = q_table[new_discrete_state + (action,)]
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_q)
            #new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    if epsilon > 0.05: #epsilon modification
        if episode_reward > prior_reward and episode % CHECKPOINT_INTERVAL==0:
            #epsilon = math.pow(epsilon_decay_value, episode - 10000)
            epsilon*=epsilon_decay_value

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    reward_record.append(episode_reward)
    
    if episode % CHECKPOINT_INTERVAL == 0: #print the average time and the average reward
        mean_reward = total_reward / CHECKPOINT_INTERVAL
        print("Mean Reward:{}".format(mean_reward))
        mean_reward_record.append(mean_reward)
        total_reward = 0

env.close()

# plt.plot(range(len(reward_record)),reward_record)
# plt.xlabel('Episodes')
# plt.ylabel('Score')
# plt.title('SARSA in CartPole-v1')
# plt.savefig('sarsa_cartpole.png')
# plt.show()

plt.plot(range(len(mean_reward_record)),mean_reward_record)
plt.xlabel('Every {} Episodes'.format(CHECKPOINT_INTERVAL))
plt.ylabel('Mean Score')
plt.title('SARSA in CartPole-v1 (Checkpoint Interval {})'.format(CHECKPOINT_INTERVAL))
plt.savefig('mean_sarsa_cartpole.png')
plt.show()