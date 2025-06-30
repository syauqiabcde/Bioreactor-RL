from model_bioreactor import bioreactor
import numpy as np
from gym import Env
from gym.spaces import Box

class Bioreactor(Env):
    def __init__(self):
        super(Bioreactor, self).__init__()

        # Define action space ranges
        self.action_ranges = {
            "I_red_light": (0, 100),    # in W/m2
            "I_blue_light": (0, 100),   # in W/m2
            "CO2_ratio": (0, 0.1),
            "Facid": (0, 1e-7),         # in m3/s
            "Fbase": (0, 1e-7),         # in m3/s
            "Fs": (0, 1e-4),            # in kg/s
            "Q_heat": (-4, 4)           # in kW
        }

        action_low = np.array([low for low, _ in self.action_ranges.values()], dtype=np.float32)
        action_high = np.array([high for _, high in self.action_ranges.values()], dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)

        # Define observation space ranges (In range(min,max) not optimum condition)
        self.observation_ranges = {
            "I_sunlight": (0.0, 800.0),   # in W/m2
            "CO2": (0.0, 50.0),  # in g/m3
            "pH": (8.0, 10.5),
            "x": (1.0, 100.0),  # in g/m3
            "S": (10.0, 50.0),  # in g/m3            
            "T": (15.0, 35.0),    # in C
            "T_env": (-5.0, 50.0),   # in C
        }

        observation_low = np.array([low for low, _ in self.observation_ranges.values()], dtype=np.float32)
        observation_high = np.array([high for _, high in self.observation_ranges.values()], dtype=np.float32)
        self.observation_space = Box(low=observation_low, high=observation_high, dtype=np.float32)

        # Initialize environment variables dynamically
        self.input_sequence = ['x', 
                'S',  
                'I_sunlight',
                'I_red_light',
                'I_blue_light', 
                'level',
                'Fs',
                'FCO2',
                'Fair',
                'FH2O',
                'Facid',
                'Fbase',
                'T',
                'CO2',
                'O2',
                'pH', 
                'd', 
                'z',
                'T_env',
                'RH',
                'wind_speed',
                'Q_heat']
        
        self.result_sequence = ['x', 
                'S',  
                'I_sunlight',
                'I_red_light',
                'I_blue_light', 
                'Fs',
                'CO2_ratio',
                'Facid',
                'Fbase',
                'T',
                'CO2',
                'pH', 
                'Q_heat',
                'T_env',
                'miu']

        self.d = 0.6
        self.z = 5
        self.FH2O = 0
        self.Fgas = 200/1000/60     # 200 L per minute total gas going in
        self.CO2_ratio = 0

        self.current_step = 0
        self.max_steps = 24 * 15 # 15 day step
        self.episode_reward = np.zeros(5)
        self.rewards = []
        self.result = []
        self.result_episode = []
        self.mius = ['miu', 'miu_light', 'miu_CO2', 'miu_pH', 'miu_S', 'miu_T']

    def _scale_action(self, tanh_output, min_action, max_action):
        scaled_output = (tanh_output + 1.0) / 2.0
        final_output = min_action + scaled_output * (max_action - min_action)
        return final_output
    
    def _tanh(self, value):
        return (np.exp(value) - np.exp(-value))/(np.exp(value) + np.exp(-value))
    
    def _get_result(self):
        result = [getattr(self, key) for key in self.result_sequence]
        self.result.append(result)

    def _get_obs(self):
        self.obs = np.array([getattr(self, key) for key in self.observation_ranges.keys()], dtype=np.float32)

    def _compute_reward(self):
        light_elc = (self.I_red_light + self.I_blue_light)/(self.action_ranges['I_red_light'][1]+self.action_ranges['I_blue_light'][1])     # converting the value to 0.0 - 1.0 to have same magnitude with miu
        
        reward_light = (self.miu_light  - light_elc * 0.5) * 10
        reward_CO2 = self.miu_CO2 * 10 
        reward_pH = self.miu_pH * 10
        reward_substrate = self.miu_S * 10
        reward_T = self.miu_T * 10

        return [reward_light, reward_CO2, reward_pH, reward_substrate, reward_T]
    
    def step(self, action):
        self.current_step += 1
        
        action = np.array([
            self._scale_action(action[i], low, high)
            for i, (low, high) in enumerate(self.action_ranges.values())
        ])

        for i, key in enumerate(self.action_ranges.keys()):
            setattr(self, key, action[i])

        self.CO2_ratio = 0.0
        self.Fair = (1 - self.CO2_ratio) * self.Fgas
        self.FCO2 = self.CO2_ratio * self.Fgas

        model = bioreactor(**{key: getattr(self, key) for key in self.input_sequence})
        self.x, self.S, self.level, self.T, self.CO2, self.O2, self.pH, miu = model.solve()

        for i in range(len(self.mius)):
            setattr(self, self.mius[i], miu[i])

        reward = self._compute_reward()
        self.episode_reward += np.array(reward)

        self._get_result()

        # done if self.current_steo>= self.max_steps
        done = self.current_step >= self.max_steps

        if done:
            self.result_episode.append(self.result)
            self.rewards.append(self.episode_reward)

        self._initialize_env()
        self._get_obs()
        return self.obs.copy(), reward, done, {}

    def reset(self):
        self.episode_reward = np.zeros(5)
        self.current_step = 0
        self.result = []

        for key, (param1, param2) in self.observation_ranges.items():
            value = np.random.uniform(param1, param2)
            setattr(self, key, value)

        self.level = np.random.uniform(0.2,0.8)
        self.O2 = np.random.uniform(1,10)
        self._initialize_env()

        self._get_obs()
        return self.obs.copy() 

    def render(self):
        pass

    def _initialize_env(self):
        self.I_sunlight = np.random.uniform(0, 800.0)
        self.I_sunlight = np.random.choice((0.0, self.I_sunlight), p=[0.3, 0.7])
        self.T_env = np.random.uniform(self.observation_ranges['T_env'][0], self.observation_ranges['T_env'][1])        
        self.RH = np.random.uniform(0, 1)
        self.wind_speed = np.random.uniform(0, 10)

   
