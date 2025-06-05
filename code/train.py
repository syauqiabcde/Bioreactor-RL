import numpy as np 
import tensorflow as tf
from ddpg_agent import Agent
from memory import ReplayBuffer
from environment import Bioreactor
from noise import GaussianWhiteNoiseProcess
from utils import reward_viz, avg_state_viz, result_to_xlsx
from tensorflow import keras
from keras.layers import Dense, Input, Concatenate, BatchNormalization, Activation
from keras.models import Model
import time
import os
import math
from datetime import datetime

def ActorNetwork(n_actions, n_states, activation='tanh', units = [32]):   

    i = Input(shape=(n_states,))
    x = i

    for j in range(len(units)):
        x = Dense(units=units[j],activation='tanh')(i) 

    x = Dense(units=n_actions,activation='tanh')(x)
    
    model = Model(i,x)
    return model

def CriticNetwork(n_actions, n_states, units = [16]):
    observation_input = Input(shape=(n_states,))
    action_input = Input(shape=(n_actions,))    
    merged = Concatenate()([action_input, observation_input])
    x = merged

    x = Dense(units[0], activation='relu')(x)

    for i in range(len(units)):
        x = Dense(units[i], activation='linear')(x)
    
    x = Dense(units=1, activation='linear')(x)
    
    model = Model(inputs=[action_input, observation_input], outputs=x)
    return model

if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    env = Bioreactor()
    agent_name = ['Light agent', 'CO2 agent', 'pH agent', 'Substrate agent', 'Temperature agent']
    act_lrs = [5e-6, 1e-7, 2e-6, 5e-7, 5e-5]
    crit_lrs = [6e-6, 2e-7, 4e-6, 7e-7, 8e-5]
    actor_activation = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh']
    hidden_layer = [[128, 128], [8], [256, 256], [64, 64], [256, 256]]
    critic_units = [[64,64], [8], [128, 128], [64, 64], [64,64]]
    batch_size = [128, 64, 256, 256, 128]
    num_actions = [2,   # light agent (I red light & blue light)
               1,       # CO2 agent (CO2 ratio)
               2,       # pH agent (Acid and base injection)
               1,       # Substrate agent (Substrate flow)
               1]       # Temperature agent (Q heat)

    num_states = [1,    # light agent (I sunlight)
               1,       # CO2 agent (CO2 concentration)
               1,       # pH agent (pH)
               2,       # Substrate agent (x and S)
               2]       # Temperature agent (T and T env)

    agents = {name: Agent(actor_network= ActorNetwork(num_actions[i], num_states[i], activation=actor_activation[i],units = hidden_layer[i]),
                  critic_network= CriticNetwork(num_actions[i], num_states[i], units = critic_units[i]),
                  nb_states= num_states[i], 
                  nb_actions = num_actions[i], 
                  noise = GaussianWhiteNoiseProcess(size = num_actions[i],sigma=1.), 
                  memory = ReplayBuffer(capacity=1000000), 
                  actor_lr=act_lrs[i], 
                  critic_lr=crit_lrs[i], 
                  gamma=0.99, 
                  tau=0.001, 
                  batch_size= batch_size[i])
            for i, name in enumerate(agent_name)}
    
    nb_episodes = 1000
    steps_per_episode  = env.max_steps

    total_runtime = 0
    episode = 0
    reward_history = []
    losses = np.zeros((int(steps_per_episode * nb_episodes), 2, len(agents)))
    print(f"Start running for {nb_episodes} episodes")
    print(f'   Episodes    | -------------------------   Average agents reward per step (max = 10.0)  ------------------------- |    Episodic Runtime     |')
    num_steps = 0

    for eps in range(nb_episodes):
        start = time.time()
        state = env.reset()

        states = []

        idx_start = 0
        for i in num_states:
            idx_end = idx_start + i
            states.append(state[idx_start:idx_end])
            idx_start = idx_end

        done = False
        eps_reward = np.zeros(len(agent_name))
            
        episode += 1
        steps = 0

        while not done:
            steps += 1
            actions = []

            for i, agent_name in enumerate(agents):
                state = states[i]
                action = agents[agent_name].choose_action(state, decay_rate = 0.99999, steps = num_steps)

                # if agent_name == 'CO2 agent':
                #     states[2][0] = action
                
                actions.append(action)

            action = np.concatenate(actions)
            next_state, rewards, done, info = env.step(action)
            reward_history.append(rewards)

            next_states = []
            idx_start = 0
            for i in num_states:
                idx_end = idx_start + i
                next_states.append(next_state[idx_start:idx_end])
                idx_start = idx_end

            # next_states[2][0] = actions[1]

            for i, agent_name in enumerate(agents):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                eps_reward[i] += reward
                next_state = next_states[i]
                agent = agents[agent_name]

                agent.remember(state, action, reward, next_state, done)
                actor_loss, critic_loss = agent.learn()
                num_steps = int(steps_per_episode * eps + steps)
                losses[num_steps-1,0,i] = actor_loss
                losses[num_steps-1,1,i] = critic_loss

            states = next_states
            

        end = time.time()
        runtime = (end - start)
        total_runtime += runtime
        print(f"Episode {episode}/{nb_episodes} | {' | '.join([f'{agent_name}: {eps_reward[i]/steps_per_episode:.1f}' for i, agent_name in enumerate(agents)])} | Runtime: {runtime:.2f} s/episode |")

    print(f"Total runtime {total_runtime/60:.2f} minutes")

    current_datetime = datetime.now()
    date = current_datetime.strftime("%m_%d_%Y")

    rewards_eps = np.array(env.rewards)
    result = np.array(env.result_episode)
    name = env.result_sequence
    reward_history = np.array(reward_history)
    result = result_to_xlsx(result, name, filename=f'output_{date}.xlsx')    

    for i, agent in enumerate(agents):
        reward_viz(rewards_eps[:,i], nb_episodes, rolling_window = 10, confidence = 0.95, title=agent)
        reward_viz(losses[:,0,i], int(steps_per_episode * nb_episodes), rolling_window = 10, confidence = 0.95, title=f'losses actor {agent}')
        reward_viz(losses[:,1,i], int(steps_per_episode * nb_episodes), rolling_window = 10, confidence = 0.95, title=f'losses critic {agent}')
        agents[agent].save_models(name=f'{agent}_{date}')

    np.savetxt(f'reward eps_{date}.csv', rewards_eps, delimiter=',', fmt='%f')
    avg_state_viz(result['x'], nb_episodes, steps_per_episode, eps_range=100, confidence=.95, state_name='Algae growth (g/m3)')
