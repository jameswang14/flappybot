import time
import numpy as np
from vision import FlappyBirdBot
from cnn_dqn_player import CNNDQNPlayer


class CNNDQNTrainer:
    """Trainer for CNN DQN Flappy Bird agent"""

    def __init__(
        self,
        episodes=1000,
        max_steps_per_episode=3000,
        save_interval=10,
        model_path="flappybird_cnn_dqn.pth",
        record_best=True,
    ):
        """
        Initialize CNN DQN trainer.

        Args:
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            save_interval: Save model every N episodes
            model_path: Path to save/load model
            record_best: Whether to record best episode
        """
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_interval = save_interval
        self.model_path = model_path
        self.record_best = record_best

        # Initialize bot (no debug window for faster training)
        self.bot = FlappyBirdBot(
            enable_ai=False,  # We'll control it manually
            shadow_mode=False,
            show_debug=False,
        )

        # Initialize CNN DQN player
        self.agent = CNNDQNPlayer(
            action_size=2,
            learning_rate=0.00025,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=32,
            target_update_freq=1000,
        )

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.best_reward = -float("inf")

    def calculate_reward(self, alive, prev_frame, current_frame):
        """
        Calculate reward for current state.

        Args:
            alive: Whether bird is still alive
            prev_frame: Previous frame (for detecting progress)
            current_frame: Current frame

        Returns:
            float: Reward value
        """
        if not alive:
            return -10.0  # Large penalty for dying

        # Small reward for staying alive
        reward = 0.1

        # Could add additional rewards based on frame analysis
        # For now, keep it simple - the CNN will learn from raw pixels

        return reward

    def is_bird_alive(self):
        return self.bot.bird_alive

    def train_episode(self, episode_num):
        """
        Train for one episode.

        Args:
            episode_num: Current episode number

        Returns:
            tuple: (total_reward, episode_length, avg_loss)
        """
        total_reward = 0
        episode_losses = []
        step = 0
        alive = True

        # Track bird position history to detect game over
        bird_position_history = []
        max_history_length = 20
        position_threshold = 5  # pixels - allow small movements

        print(
            f"\nEpisode {episode_num}/{self.episodes} (epsilon: {self.agent.epsilon:.4f})"
        )
        # Start game
        self.bot.start_game()

        # Wait a moment for game to start
        time.sleep(0.5)

        while step < self.max_steps_per_episode and alive:
            # Capture frame
            frame = self.bot.get_frame()

            # Also detect bird position for game-over detection
            self.bot.detect_bird(frame)
            bird_pos_tuple = self.bot.bird_position

            # Get state (frame + time since last tap)
            state_frame, state_time = self.agent.get_state(frame)

            # Select and execute action
            action = self.agent.select_action(state_frame, state_time, training=True)
            should_tap = action == 1

            if should_tap:
                tap_region = {
                    "x": self.bot.monitor["left"],
                    "y": self.bot.monitor["top"],
                    "width": self.bot.monitor["width"],
                    "height": self.bot.monitor["height"],
                }
                self.agent.execute_tap(tap_region)

            # Small delay to let action take effect
            time.sleep(0.03)

            # Get next state
            next_frame = self.bot.get_frame()
            next_state_frame, next_state_time = self.agent.get_state(next_frame)

            # Detect bird for game-over detection
            self.bot.detect_bird(next_frame)
            next_bird_pos_tuple = self.bot.bird_position

            # Track bird position to detect game over (bird hasn't moved)
            if next_bird_pos_tuple:
                bird_x, bird_y, bird_w, bird_h = next_bird_pos_tuple
                bird_position_history.append((bird_x, bird_y))

                # Keep only the last max_history_length positions
                if len(bird_position_history) > max_history_length:
                    bird_position_history.pop(0)

                # Check if bird hasn't moved in the last 20 frames
                if len(bird_position_history) == max_history_length:
                    # Calculate max distance from first position in history
                    first_x, first_y = bird_position_history[0]
                    max_distance = 0
                    for hist_x, hist_y in bird_position_history:
                        distance = (
                            (hist_x - first_x) ** 2 + (hist_y - first_y) ** 2
                        ) ** 0.5
                        max_distance = max(max_distance, distance)

                    # If bird hasn't moved more than threshold, assume game over
                    if max_distance < position_threshold:
                        print(
                            f"\n  Bird position hasn't changed in {max_history_length} frames - game over detected!"
                        )
                        alive = False

            # Calculate reward
            reward = self.calculate_reward(alive, frame, next_frame)
            total_reward += reward

            # Store transition
            self.agent.store_transition(
                state_frame, state_time, action, reward, next_state_frame, next_state_time, not alive
            )

            # Train
            loss = self.agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            step += 1

            # Print progress every 50 steps
            if step % 50 == 0:
                bird_status = "detected" if bird_pos_tuple else "not detected"
                print(
                    f"  Step {step}: Reward={total_reward:.2f}, Alive={alive}, Bird={bird_status}"
                )

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        print(
            f"Episode {episode_num} finished: Steps={step}, Reward={total_reward:.2f}, Loss={avg_loss:.4f}"
        )

        return total_reward, step, avg_loss

    def train(self, resume_from=None):
        """
        Train the CNN DQN agent.

        Args:
            resume_from: Path to model checkpoint to resume from
        """
        start_episode = 1

        if resume_from:
            self.agent.load(resume_from)

            # Extract episode number from filename if present
            # Expected format: flappybird_cnn_dqn_ep100.pth
            import re

            match = re.search(r"_ep(\d+)\.pth", resume_from)
            if match:
                start_episode = int(match.group(1)) + 1
                print(f"Resuming from episode {start_episode - 1}")
            else:
                print(f"Resuming training (episode number not found in filename)")

        print(f"\n{'='*60}")
        print(f"Starting CNN DQN Training")
        print(f"Episodes: {start_episode} to {self.episodes}")
        print(f"Max steps per episode: {self.max_steps_per_episode}")
        print(f"Save interval: {self.save_interval}")
        print(f"{'='*60}\n")

        for episode in range(start_episode, self.episodes + 1):
            # Train one episode
            reward, length, loss = self.train_episode(episode)

            # Track metrics
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            if loss > 0:
                self.losses.append(loss)

            # Save best model
            if reward > self.best_reward:
                self.best_reward = reward
                best_model_path = self.model_path.replace(".pth", "_best.pth")
                self.agent.save(best_model_path)
                print(f"  New best reward! Saved to {best_model_path}")

            # Save checkpoint
            if episode % self.save_interval == 0:
                checkpoint_path = self.model_path.replace(".pth", f"_ep{episode}.pth")
                self.agent.save(checkpoint_path)

            # Print statistics
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_loss = np.mean(self.losses[-10:]) if self.losses[-10:] else 0
                print(f"\n{'='*60}")
                print(f"Episode {episode} - Last 10 episodes stats:")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.2f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {self.agent.epsilon:.4f}")
                print(f"  Best Reward: {self.best_reward:.2f}")
                print(f"{'='*60}\n")

            # Wait a bit before starting next episode
            time.sleep(2)

        # Save final model
        self.agent.save(self.model_path)
        print(f"\nTraining complete! Final model saved to {self.model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN DQN for Flappy Bird")
    parser.add_argument(
        "--model-path",
        type=str,
        default="flappybird_cnn_dqn.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to train",
    )

    args = parser.parse_args()

    trainer = CNNDQNTrainer(
        model_path=args.model_path,
        episodes=args.episodes,
    )

    trainer.train(resume_from=args.resume)
