import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import pyautogui
import cv2


class CNNDQNetwork(nn.Module):
    """CNN-based Deep Q-Network for Flappy Bird using raw frames"""

    def __init__(self, action_size=2):
        """
        Initialize the CNN network.

        Args:
            action_size: Number of possible actions (tap or no-tap)
        """
        super(CNNDQNetwork, self).__init__()

        # Input: grayscale frame (1 channel) of size 84x84
        # Conv layers to extract spatial features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate size after convolutions
        # 84 -> (84-8)/4 + 1 = 20
        # 20 -> (20-4)/2 + 1 = 9
        # 9 -> (9-3)/1 + 1 = 7
        # So output is 64 * 7 * 7 = 3136

        self.fc1 = nn.Linear(3136 + 1, 512)  # +1 for time_since_last_tap feature
        self.fc2 = nn.Linear(512, action_size)

        self.relu = nn.ReLU()

    def forward(self, frame, time_feature):
        """
        Forward pass through the network

        Args:
            frame: Preprocessed frame tensor (batch_size, 1, 84, 84)
            time_feature: Time since last tap (batch_size, 1)
        """
        # Process frame through CNN
        x = self.relu(self.conv1(frame))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten CNN output
        x = x.view(x.size(0), -1)

        # Concatenate with time feature
        x = torch.cat([x, time_feature], dim=1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

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

    def push(self, state_frame, state_time, action, reward, next_frame, next_time, done):
        """Add a transition to the buffer"""
        self.buffer.append(
            (state_frame, state_time, action, reward, next_frame, next_time, done)
        )

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        (
            state_frames,
            state_times,
            actions,
            rewards,
            next_frames,
            next_times,
            dones,
        ) = zip(*batch)
        return (
            np.array(state_frames),
            np.array(state_times),
            np.array(actions),
            np.array(rewards),
            np.array(next_frames),
            np.array(next_times),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class CNNDQNPlayer:
    """CNN-based DQN player for Flappy Bird using raw frames"""

    def __init__(
        self,
        action_size=2,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=1000,
    ):
        """
        Initialize CNN DQN player.

        Args:
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
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CNN DQN using device: {self.device}")

        # Networks
        self.policy_net = CNNDQNetwork(action_size).to(self.device)
        self.target_net = CNNDQNetwork(action_size).to(self.device)
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

    def preprocess_frame(self, frame):
        """
        Preprocess frame for CNN input.

        Args:
            frame: Raw frame from game (BGR format)

        Returns:
            Preprocessed frame (1, 84, 84) numpy array
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add channel dimension (1, 84, 84)
        processed = np.expand_dims(normalized, axis=0)

        return processed

    def get_state(self, frame):
        """
        Extract state from raw frame.

        Args:
            frame: Raw frame from game

        Returns:
            tuple: (preprocessed_frame, time_since_last_tap)
        """
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)

        # Time since last tap (normalized to 1.0 second)
        current_time = time.time()
        time_since_tap = current_time - self.last_tap_time
        # Normalize: cap at 1.0 second
        normalized_time_since_tap = min(time_since_tap / 1.0, 1.0)

        return processed_frame, normalized_time_since_tap

    def select_action(self, state_frame, state_time, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state_frame: Preprocessed frame
            state_time: Time since last tap
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
                frame_tensor = (
                    torch.FloatTensor(state_frame).unsqueeze(0).to(self.device)
                )
                time_tensor = (
                    torch.FloatTensor([[state_time]]).to(self.device)
                )
                q_values = self.policy_net(frame_tensor, time_tensor)
                return q_values.argmax().item()

    def store_transition(
        self, state_frame, state_time, action, reward, next_frame, next_time, done
    ):
        """Store a transition in replay buffer"""
        self.memory.push(
            state_frame, state_time, action, reward, next_frame, next_time, done
        )

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from replay buffer
        (
            state_frames,
            state_times,
            actions,
            rewards,
            next_frames,
            next_times,
            dones,
        ) = self.memory.sample(self.batch_size)

        # Convert to tensors
        state_frames = torch.FloatTensor(state_frames).to(self.device)
        state_times = torch.FloatTensor(state_times).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_frames = torch.FloatTensor(next_frames).to(self.device)
        next_times = torch.FloatTensor(next_times).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(state_frames, state_times).gather(
            1, actions.unsqueeze(1)
        )

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_frames, next_times).max(1)[0]
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

    def decide_action(self, frame):
        """
        Decide whether to tap based on current frame.

        Args:
            frame: Raw frame from game

        Returns:
            bool: True if should tap, False otherwise
        """
        state_frame, state_time = self.get_state(frame)
        action = self.select_action(state_frame, state_time, training=False)
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

    def play_step(self, frame, tap_region, shadow_mode=False):
        """
        Execute one step of gameplay.

        Args:
            frame: Raw frame from game
            tap_region: Dict with screen coordinates
            shadow_mode: If True, don't actually tap

        Returns:
            bool: True if would tap (or did tap), False otherwise
        """
        should_tap = self.decide_action(frame)

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
        print(f"CNN DQN model saved to {filepath}")

    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps = checkpoint.get("steps", 0)
        print(f"CNN DQN model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.4f}, Steps: {self.steps}")
