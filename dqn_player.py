import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import pyautogui


class DQNetwork(nn.Module):
    """Deep Q-Network for Flappy Bird"""

    def __init__(self, state_size=7, action_size=2, hidden_size=128):
        """
        Initialize the neural network.

        Args:
            state_size: Number of state features (default 7)
            action_size: Number of possible actions (tap or no-tap)
            hidden_size: Size of hidden layers
        """
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""

    def __init__(self, capacity=10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNPlayer:
    """DQN-based player for Flappy Bird"""

    def __init__(
        self,
        state_size=7,
        action_size=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
    ):
        """
        Initialize DQN player.

        Args:
            state_size: Number of state features (default 7)
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Rate of epsilon decay
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Training tracking
        self.steps = 0
        self.last_tap_time = 0
        self.min_tap_interval = 0.15

    def get_state(self, bird_pos, pipes, ground_y, frame_height):
        """
        Extract state features from game state.

        Args:
            bird_pos: Dict with bird position
            pipes: List of pipe dicts
            ground_y: Y coordinate of ground
            frame_height: Height of game frame

        Returns:
            numpy array of normalized state features (7 features)
        """
        if not bird_pos:
            return np.zeros(self.state_size)

        bird_y = bird_pos.get("center_y", bird_pos.get("y", 0))
        bird_x = bird_pos.get("center_x", bird_pos.get("x", 0))

        # Find next pipe
        next_pipe = None
        for pipe in pipes:
            if pipe["x"] + pipe["width"] > bird_x - 50:
                next_pipe = pipe
                break

        if next_pipe:
            # Horizontal distance to pipe
            pipe_dist = next_pipe["x"] - bird_x
            # Vertical distance to gap center
            gap_center = next_pipe.get("gap_center", frame_height / 2)
            vertical_dist = bird_y - gap_center
            # Gap size
            gap_size = next_pipe.get("gap_height", 100)
            # Pipe top and bottom
            pipe_top = next_pipe.get("top_pipe", {}).get("y", 0) if next_pipe.get("top_pipe") else 0
            pipe_bottom = next_pipe.get("bottom_pipe", {}).get("y", frame_height) if next_pipe.get("bottom_pipe") else frame_height
        else:
            pipe_dist = frame_height
            vertical_dist = 0
            gap_size = 100
            pipe_top = 0
            pipe_bottom = frame_height

        # Distance to ground
        ground_dist = ground_y - bird_y if ground_y else frame_height - bird_y

        # Time since last tap (normalized to 1.0 second)
        current_time = time.time()
        time_since_tap = current_time - self.last_tap_time
        # Normalize: cap at 1.0 second (anything beyond 1 second is considered "ready to tap")
        normalized_time_since_tap = min(time_since_tap / 1.0, 1.0)

        # Normalize features (important for neural network)
        state = np.array(
            [
                bird_y / frame_height,  # Normalized bird Y position (0-1)
                vertical_dist / frame_height,  # Normalized vertical distance to gap
                pipe_dist / frame_height,  # Normalized horizontal distance to pipe
                gap_size / frame_height,  # Normalized gap size
                ground_dist / frame_height,  # Normalized distance to ground
                (pipe_bottom - pipe_top) / frame_height,  # Normalized pipe opening
                normalized_time_since_tap,  # Time since last tap (0-1, capped at 1 second)
            ],
            dtype=np.float32,
        )

        return state

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)

        Returns:
            0 (no tap) or 1 (tap)
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.action_size - 1)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def decide_action(self, bird_pos, pipes, ground_y=None, frame_height=600):
        """
        Decide whether to tap based on current state.

        Args:
            bird_pos: Dict with bird position
            pipes: List of pipe dicts
            ground_y: Y coordinate of ground
            frame_height: Height of game frame

        Returns:
            bool: True if should tap, False otherwise
        """
        state = self.get_state(bird_pos, pipes, ground_y, frame_height)
        action = self.select_action(state, training=True)
        return action == 1  # Return True if action is tap

    def execute_tap(self, tap_region):
        """
        Execute a screen tap within the specified region.

        Args:
            tap_region: Dict with screen coordinates
        """
        current_time = time.time()

        # Prevent tapping too frequently
        if current_time - self.last_tap_time < self.min_tap_interval:
            return

        # Calculate tap position
        tap_x = tap_region["x"] + tap_region["width"] // 2
        tap_y = tap_region["y"] + tap_region["height"] // 2

        # Execute the tap
        pyautogui.click(tap_x, tap_y)
        self.last_tap_time = current_time

    def play_step(
        self, bird_pos, pipes, tap_region, shadow_mode=False, ground_y=None, frame_height=600
    ):
        """
        Execute one step of gameplay.

        Args:
            bird_pos: Dict with bird position
            pipes: List of pipe dicts
            tap_region: Dict with screen coordinates
            shadow_mode: If True, don't actually tap
            ground_y: Y coordinate of ground
            frame_height: Height of game frame

        Returns:
            bool: True if would tap (or did tap), False otherwise
        """
        should_tap = self.decide_action(bird_pos, pipes, ground_y, frame_height)

        if should_tap and not shadow_mode:
            self.execute_tap(tap_region)

        return should_tap

    def save(self, filepath):
        """Save model weights"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps = checkpoint.get("steps", 0)
        print(f"Model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.4f}, Steps: {self.steps}")
