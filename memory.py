import numpy as np

# class ReplayBuffer:
#     def __init__(self, max_size, input_shape, n_actions):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size, *input_shape))
#         self.new_state_memory = np.zeros((self.mem_size, *input_shape))
#         self.action_memory = np.zeros((self.mem_size, n_actions))
#         self.reward_memory = np.zeros(self.mem_size)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

#     def store_transition(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size

#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.terminal_memory[index] = done

#         self.mem_cntr += 1

#     def sample_buffer(self, batch_size):
#         max_mem = min(self.mem_cntr, self.mem_size)

#         batch = np.random.choice(max_mem, batch_size, replace=False)

#         states = self.state_memory[batch]
#         states_ = self.new_state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         dones = self.terminal_memory[batch]

#         return states, actions, rewards, states_, dones
    
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.n_stored = 0

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idx:
            s, a, r, s2, d = self.memory[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s2)
            dones.append(d)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
