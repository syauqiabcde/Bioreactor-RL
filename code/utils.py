import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pandas as pd
from tensorflow import keras
import os
from environment import Bioreactor
import tensorflow as tf
import torch

def reward_viz(rewards, nb_episode, rolling_window = 10, confidence = 0.95, title=''):
    fig = plt.figure(dpi=400)
    ax = fig.add_axes([0,0,1,1])
    x_axis = np.arange(1, nb_episode+1)

    rolling_avg = np.convolve(rewards, np.ones(rolling_window)/rolling_window, mode='valid')
    std_devs = [np.std(rewards[max(0, i-rolling_window+1):i+1]) for i in range(len(rewards))]
    std_devs = np.array(std_devs[len(std_devs) - len(rolling_avg):])  # Align sizes
    margin_of_error = st.t.ppf((1 + confidence) / 2, df=rolling_window-1) * std_devs / np.sqrt(rolling_window)

    ax.plot(x_axis[:len(rolling_avg)], rolling_avg, label=f"Rolling Avg ({rolling_window} episodes)", color="red", linewidth=2)
    ax.fill_between(x_axis[:len(rolling_avg)], rolling_avg - margin_of_error, rolling_avg + margin_of_error, color="red", alpha=0.2, label="95% CI")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.set_title(title)

def avg_state_viz(state_array, nb_episodes, steps_per_episode, eps_range=100, confidence=.95, state_name='state'):
    state_array = state_array.T
    early = np.average(state_array[:,:eps_range], axis=1)
    early_moe = np.std(state_array[:,:eps_range], axis=1) * st.norm.ppf(confidence) / eps_range
    last = np.average(state_array[:,nb_episodes-eps_range:], axis=1)
    last_moe = np.std(state_array[:,nb_episodes-eps_range:], axis=1) * st.norm.ppf(confidence) / eps_range
    x_axis = np.arange(0, steps_per_episode)

    fig = plt.figure(dpi=400)
    ax = fig.add_axes([0,0,1,1])

    ax.plot(x_axis, early, label=f'{state_name} of first {eps_range} episode', color='red', linewidth = 2)
    ax.fill_between(x_axis, early + early_moe, early - early_moe,  color="red", alpha=0.2)
    ax.plot(x_axis, last, label=f'{state_name} of last {eps_range} episode', color='blue', linewidth = 2)
    ax.fill_between(x_axis, last + last_moe, last - last_moe,  color="blue", alpha=0.2)
    ax.legend()
    ax.set_xlabel('Steps')
    ax.set_ylabel(f'{state_name}')

def result_to_xlsx(result, name, filename="output.xlsx"):
    result_dict = {}

    for i in range(result.shape[-1]):
        result_dict[name[i]] = result[:,:,i]

    with pd.ExcelWriter(filename) as writer:
        for sheet_name, array in result_dict.items():
            array = array.T
            df = pd.DataFrame(array)
            df.to_excel(writer, sheet_name=sheet_name, index=True, header=True)

    print("Excel file created successfully!")
    return result_dict

class ControlAgent:
    def __init__(self):       
        ###########################################
        #        Loading agents for MARL          #
        ###########################################
        def custom_activation(x):
            return tf.clip_by_value(x, -1.0, 1.0)
    
        base_dir = r'D:\Ahmad Syauqis Document\Paper - RL for microalgae\UI\models'

        agent_filenames = {
            key: os.path.join(base_dir, f"{key}.h5")
            for key in ['light agent', 'CO2 agent', 'pH agent', 'substrate agent', 'temperature agent']
        }

        self.MARL_agents = {key: keras.models.load_model(path, custom_objects={'custom_activation': custom_activation}) for key, path in agent_filenames.items()}
        self.num_states = [1,    # light agent (I sunlight)
                        1,       # CO2 agent (CO2 concentration)
                        1,       # pH agent (pH)
                        2,       # Substrate agent (x and S)
                        2]       # Temperature agent (T and T env)
        
        self.env = Bioreactor()
        action_ranges = self.env.action_ranges 
        self.action_ranges_list = [(min, max) for (min, max) in action_ranges.values()]
        
        ###########################################
        #        Loading agents for HMARL         #
        ###########################################

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        actor_names = ['Light', 'CO₂', 'pH', 'Substrate', 'Temperature']
        self.agent_hmarl = {}
        self.num_states_hmarl = [1, 1, 2, 2, 2] # from training HMARL

        for name in actor_names:
            path = os.path.join(base_dir, f'actor_{name}.h5')
            model = torch.load(path, map_location=self.device)
            model.eval()
            self.agent_hmarl[name] = model

        mgr_path = os.path.join(base_dir, 'manager.h5')
        self.manager_hmarl = torch.load(mgr_path, map_location=self.device)
        self.manager_hmarl.eval()

        self.env_hmarl = Bioreactor()
        action_ranges_hmarl = self.env_hmarl.action_ranges
        self.action_ranges_list_hmarl = [ (min, max) for (min, max) in action_ranges_hmarl.values() ]

        ###########################################
        # General config for conventional control #
        ###########################################

        self.temp_state = 0  # 0 = off, 1 = heating, -1 = cooling
        self.t_low = 20
        self.t_high = 35
        self.t_opt = 30
        self.ph_state = 0    # 0 = off, 1 = base inject, -1 = acid inject
        self.ph_low = 8.5
        self.ph_high = 10
        self.ph_opt = 9.2
        self.s_state = 0
        self.s_low = 50
        self.s_high = 200

    def temperature_control(self, current_temp):
        if self.temp_state == 0 and current_temp <= self.t_low:
            self.temp_state = 1
        elif self.temp_state == 1 and current_temp >= self.t_opt:
            self.temp_state = 0
        if self.temp_state == 0 and current_temp >= self.t_high:
            self.temp_state = -1
        elif self.temp_state == -1 and current_temp <= self.t_opt:
            self.temp_state = 0

        if self.temp_state == 1:
            return [self.action_ranges_list[-1][1]]
        elif self.temp_state == -1:
            return [self.action_ranges_list[-1][0]]
        else:
            return [0]
        
    def ph_control(self, current_ph):
        if self.ph_state == 0 and current_ph <= self.ph_low:
            self.ph_state = 1  # add base
        elif self.ph_state == 1 and current_ph >= self.ph_opt:
            self.ph_state = 0  # stop base

        if self.ph_state == 0 and current_ph >= self.ph_high:
            self.ph_state = -1  # add acid
        elif self.ph_state == -1 and current_ph <= self.ph_opt:
            self.ph_state = 0  # stop acid

        if self.ph_state == 1:
            return [0.0, self.action_ranges_list[4][1]]
        elif self.ph_state == -1:
            return [self.action_ranges_list[3][1], 0.0]
        else:
            return [0.0, 0.0]
        
    def substrate_control(self, current_substrate):
        if self.s_state == 0 and current_substrate <= self.s_low:
            self.s_state = 1  # add substrate
        elif self.s_state == 1 and current_substrate >= self.s_high:
            self.s_state = 0  # stop substrate

        if self.s_state == 1:
            return [self.action_ranges_list[-2][1]]
        else:
            return [0.0]

    def take_action(self, obs, mode):
        actions = []

        if mode == 'MARL':
            idx_state = 0
            idx_action = 0

            for i, (key, agent) in enumerate(self.MARL_agents.items()):
                idx_end = idx_state + self.num_states[i]
                states = obs[idx_state:idx_end]
                states = np.array(states).reshape(1, -1)
                action = agent.predict(states, verbose=0)

                for j, act in enumerate(action):
                    min_a, max_a = self.action_ranges_list[idx_action + j]
                    scaled_action = self.env._scale_action(act, min_a, max_a)
                    actions.append(scaled_action)
        
                idx_action += action.shape[1]
                idx_state = idx_end
            return actions
        
        elif mode == 'HMARL':
            gs = torch.from_numpy(obs.astype(np.float32))\
                      .unsqueeze(0)\
                      .to(self.device)

            with torch.no_grad():

                raw_scores = self.manager_hmarl(gs)      # tensor shape [1,5]
            w = torch.softmax(raw_scores, dim=-1)        # [1,5]
            w = w.cpu().numpy().flatten()                # array length 5

            idx_s = idx_a = 0
            
            for i, actor in enumerate(self.agent_hmarl.values()):
                n_s = self.num_states_hmarl[i]
                lo = torch.from_numpy(
                         obs[idx_s:idx_s + n_s].astype(np.float32)
                     ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    raw = actor(lo).cpu().numpy().flatten() 

                for j, val in enumerate(raw):
                    mn, mx = self.action_ranges_list_hmarl[idx_a + j]
                    scaled = self.env_hmarl._scale_action(val, mn, mx)
                    print(i, scaled)
                    actions.append(w[i] * scaled)

                idx_s += n_s
                idx_a += raw.shape[0]

            return actions

        
        elif mode == 'conventional':
            # Light control
            actions.append(np.array([100.0, 100.0]) if obs[0] <= 50 else np.array([0.0, 0.0]))

            # CO2 control
            actions.append(np.array([0.0]))

            # pH control
            ph_action = self.ph_control(obs[2])
            actions.append(np.array(ph_action))

            # Substrate control 
            s_action = self.substrate_control(obs[4])
            actions.append(np.array(s_action))

            # Temperature control
            temp_action = self.temperature_control(obs[-2])
            actions.append(np.array(temp_action))

            return actions