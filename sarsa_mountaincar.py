import numpy as np 
import gym 
import math 
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
#print(env.action_space.n)

LEARNING_RATE = 0.1

DISCOUNT = 0.9
EPISODES = 100000
CHECKPOINT_INTERVAL=1000
total_reward = 0
prior_reward = 0

POS_GRID=10
VELO_GRID=20

pos_range=np.linspace(-1.2,0.6,num=POS_GRID)
velo_range=np.linspace(-0.07,0.07,num=VELO_GRID)

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING=EPISODES//2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-1, high=0, size=([POS_GRID,VELO_GRID] + [env.action_space.n]))

def get_discrete_state(state):
    pos,velo=state
    pos_num=np.digitize(pos,pos_range)
    velo_num=np.digitize(velo,velo_range)
    return (pos_num,velo_num)

reward_record=[]
mean_reward_record=[]



for episode in range(EPISODES+1): #go through the episodes

    discrete_state = get_discrete_state(env.reset()) #get the discrete start for the restarted environment 
    done = False
    episode_reward = 0 #reward starts as 0 for the start of the episode

    if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state]) 
    else:
        action = np.random.randint(0, env.action_space.n) #a random action

    new_state, reward, done, _ = env.step(action) #take action to get new states, reward, and the "done" status.
    new_discrete_state = get_discrete_state(new_state)
    episode_reward += reward #add the reward

    if episode!=0 and episode % CHECKPOINT_INTERVAL == 0: 
        print("Episode: " + str(episode))


    while not done: 
        
        if np.random.random() > epsilon:
            next_action = np.argmax(q_table[new_discrete_state]) 
        else:
            next_action = np.random.randint(0, env.action_space.n) #a random action


        if episode % CHECKPOINT_INTERVAL == 0: #render
            env.render()

        if not done: #update q-table
            future_q = q_table[new_discrete_state + (next_action,)]
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state
        
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) 
        else:
            action = np.random.randint(0, env.action_space.n) #a random action

        new_state, reward, done, _ = env.step(action) #take action to get new states, reward, and the "done" status.
        new_discrete_state = get_discrete_state(new_state)
        episode_reward += reward #add the reward


        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value


    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    reward_record.append(episode_reward)
    
    if episode!=0 and episode % CHECKPOINT_INTERVAL == 0: #print the average time and the average reward
        mean_reward = total_reward / CHECKPOINT_INTERVAL
        print("Mean Reward:{}".format(mean_reward))
        mean_reward_record.append(mean_reward)
        total_reward = 0

env.close()

plt.plot(range(len(mean_reward_record)),mean_reward_record)
plt.xlabel('Every {} Episodes'.format(CHECKPOINT_INTERVAL))
plt.ylabel('Mean Score')
plt.title('SARSA in MountainCar-v0 (Checkpoint Interval {})'.format(CHECKPOINT_INTERVAL))
plt.savefig('mean_sarsa_mountaincar.png')
plt.show()