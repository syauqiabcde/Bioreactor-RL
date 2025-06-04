import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Dense, Input, Concatenate
from keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

class Agent:
    def __init__(self, actor_network, critic_network, nb_states, nb_actions, noise, memory, 
                 actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.batch_size = batch_size
        self.n_states = nb_states
        self.n_actions = nb_actions
        self.noise = noise
        self.max_action = 1.0             # by default the actor network final activation function is tanh
        self.min_action = -1.0            # by default the actor network final activation function is tanh

        self.actor = actor_network
        self.critic = critic_network
        
        self.target_actor = clone_model(self.actor)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic = clone_model(self.critic)
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.update_network_parameters(tau=1)

    def reset_noise(self):
        self.noise.reset()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, name='env'):
        print('... saving models ...')
        output_dir = 'saved models'
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
        actor_path = os.path.join(output_dir, f'actor_{name}.h5')
        critic_path = os.path.join(output_dir, f'critic_{name}.h5')
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        print('Successfully save models')

    def choose_action(self, observation, noise=True, decay_rate = 1.0, steps=0.0):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        action = self.actor(state)
        
        if noise:
            action += (self.noise.sample() * decay_rate**steps)

        action = np.clip(action, self.min_action, self.max_action)[0]
        return action

    def learn(self):
        if len(self.memory.memory) < self.batch_size:
            return 0.0, 0.0

        state, action, reward, new_state, done = \
            self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        dones = tf.convert_to_tensor(done, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic([target_actions, states_]), 1)
            critic_value = tf.squeeze(self.critic([actions, states]), 1)
            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic([new_policy_actions, states])
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
        return critic_loss, actor_loss
