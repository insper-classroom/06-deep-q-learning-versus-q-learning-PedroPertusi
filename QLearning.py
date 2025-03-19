import numpy as np
import gymnasium as gym
import os

class QLearningAgent:
    def __init__(self, 
                 env: gym.Env,
                 alpha=0.1, 
                 gamma=0.99,
                 epsilon=1.0, 
                 epsilon_decay=0.99, 
                 epsilon_min=0.01):
        
        self.env = env
        
        # Discretiza o espaço de estados
        self.num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1  # dimensão: [n_pos, n_vel]
        
        self.n_actions = env.action_space.n
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table com 3 dimensões: posição, velocidade, ação
        self.q_table = np.zeros((self.num_states[0], self.num_states[1], self.n_actions))
        
        # Lista para armazenar o reward acumulado por episódio
        self.rewards_per_episode = []
        
    def transform_state(self, state):
        state_adj = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state_adj, 0).astype(int)
    
    def select_action(self, state_adj):
        if np.random.random() < 1 - self.epsilon:
            return np.argmax(self.q_table[state_adj[0], state_adj[1]])
        return np.random.randint(0, self.env.action_space.n)
    
    def train(self, num_episodes=5000, max_steps_per_episode=1000):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state_adj = self.transform_state(state)
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                action = self.select_action(state_adj)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state_adj = self.transform_state(next_state)
                
                # Atualiza a Q-table utilizando a equação de Q-Learning em 1 linha (Equação de Bellman)
                self.q_table[state_adj[0], state_adj[1], action] += self.alpha * (
                    reward + self.gamma * np.max(self.q_table[next_state_adj[0], next_state_adj[1]]) - 
                    self.q_table[state_adj[0], state_adj[1], action])
                
                state_adj = next_state_adj
                total_reward += reward
                
                if done: #or truncated (tirar o truncated para não parar no -200):
                    break
            
            self.rewards_per_episode.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_per_episode[-100:])
                print(f"Episódio: {episode+1}/{num_episodes} | Recompensa média dos últimos 100: {avg_reward:.3f} | ε: {self.epsilon:.3f}")
            
        return self.q_table
    
    def predict(self, max_steps=1000):
        state, _ = self.env.reset()
        state_adj = self.transform_state(state)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = np.argmax(self.q_table[state_adj[0], state_adj[1]])
            next_state, reward, done, truncated, _ = self.env.step(action)
            state_adj = self.transform_state(next_state)
            total_reward += reward
            steps += 1
            if done or truncated:
                break
        return total_reward, steps
    
    def get_q_table(self):
        return self.q_table
    
    def get_rewards_per_episode(self):
        return self.rewards_per_episode

    def save_q_table(self, filename):
        folder = "qtable"
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename + ".npy")
        np.save(path, self.q_table)
        print(f"Q-table salva em '{path}'.")

    def load_q_table(self, filename):
        folder = "qtable"
        path = os.path.join(folder, filename + ".npy")
        self.q_table = np.load(path)
        print(f"Q-table carregada de '{path}'.")

    