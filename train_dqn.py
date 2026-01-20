import time
import numpy as np
from vision import FlappyBirdBot
from dqn_player import DQNPlayer


class DQNTrainer:
    """Trainer for DQN Flappy Bird agent"""

    def __init__(
        self,
        episodes=1000,
        max_steps_per_episode=3000,
        save_interval=10,
        model_path="flappybird_dqn.pth",
        record_best=True,
    ):
        """
        Initialize DQN trainer.

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

        # Get frame height for state normalization
        frame = self.bot.get_frame()
        self.frame_height = frame.shape[0]

        # Initialize DQN player
        self.agent = DQNPlayer(
            state_size=7,
            action_size=2,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=50,
        )

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.best_reward = -float("inf")

    def calculate_reward(self, bird_pos, pipes, ground_y, alive, prev_bird_y=None):
        """
        Calculate reward for current state.

        Args:
            bird_pos: Dict with bird position
            pipes: List of pipe dicts
            ground_y: Y coordinate of ground
            alive: Whether bird is still alive
            prev_bird_y: Previous bird Y position

        Returns:
            float: Reward value
        """
        if not alive:
            return -10.0  # Large penalty for dying

        if not bird_pos:
            return -1.0

        bird_y = bird_pos.get("center_y", bird_pos.get("y", 0))
        bird_x = bird_pos.get("center_x", bird_pos.get("x", 0))

        reward = 0.1  # Small reward for staying alive

        # Find next pipe
        next_pipe = None
        for pipe in pipes:
            if pipe["x"] + pipe["width"] > bird_x - 50:
                next_pipe = pipe
                break

        if next_pipe:
            # Check if we passed a pipe (pipe is now behind us)
            if next_pipe["x"] + next_pipe["width"] < bird_x:
                reward += 5.0  # Large reward for passing pipe

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
        prev_bird_y = None
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

        while step < self.max_steps_per_episode and alive:
            # Capture frame and detect game elements
            frame = self.bot.get_frame()
            self.bot.detect_bird(frame)
            self.bot.detect_pipes(frame)
            self.bot.detect_ground(frame)

            # Get current state
            bird_pos_tuple = self.bot.bird_position
            if bird_pos_tuple:
                bird_x, bird_y, bird_w, bird_h = bird_pos_tuple
                bird_pos = {
                    "x": bird_x,
                    "y": bird_y,
                    "center_x": bird_x,
                    "center_y": bird_y,
                    "width": bird_w,
                    "height": bird_h,
                }
            else:
                bird_pos = None

            pipes = self.bot.pipes
            ground_y = self.bot.ground_y

            # Get state
            state = self.agent.get_state(bird_pos, pipes, ground_y, self.frame_height)

            # Select and execute action
            action = self.agent.select_action(state, training=True)
            should_tap = action == 1

            if should_tap:
                tap_region = {
                    "x": self.bot.monitor["left"],
                    "y": self.bot.monitor["top"],
                    "width": self.bot.monitor["width"],
                    "height": self.bot.monitor["height"],
                }
                self.agent.execute_tap(tap_region)

            # Get next state
            frame = self.bot.get_frame()
            self.bot.detect_bird(frame)
            self.bot.detect_pipes(frame)
            self.bot.detect_ground(frame)

            bird_pos_tuple = self.bot.bird_position
            if bird_pos_tuple:
                bird_x, bird_y, bird_w, bird_h = bird_pos_tuple
                next_bird_pos = {
                    "x": bird_x,
                    "y": bird_y,
                    "center_x": bird_x,
                    "center_y": bird_y,
                    "width": bird_w,
                    "height": bird_h,
                }
            else:
                next_bird_pos = None

            next_pipes = self.bot.pipes
            next_ground_y = self.bot.ground_y
            next_state = self.agent.get_state(
                next_bird_pos, next_pipes, next_ground_y, self.frame_height
            )

            # Track bird position to detect game over (bird hasn't moved)
            if next_bird_pos:
                bird_x = next_bird_pos.get("center_x")
                bird_y = next_bird_pos.get("center_y")
                bird_position_history.append((bird_x, bird_y))

                # Keep only the last max_history_length positions
                if len(bird_position_history) > max_history_length:
                    bird_position_history.pop(0)

                # Check if bird hasn't moved in the last 30 frames
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
            reward = self.calculate_reward(
                next_bird_pos, next_pipes, next_ground_y, alive, prev_bird_y
            )
            total_reward += reward

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, not alive)

            # Train
            loss = self.agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            # Update tracking
            if next_bird_pos:
                prev_bird_y = next_bird_pos.get("center_y")

            step += 1

            # Print progress every 100 steps
            if step % 50 == 0:
                print(
                    f"  Step {step}: Reward={total_reward:.2f}, Alive={alive}, Pipes={len(pipes)}"
                )

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        print(
            f"Episode {episode_num} finished: Steps={step}, Reward={total_reward:.2f}, Loss={avg_loss:.4f}"
        )

        return total_reward, step, avg_loss

    def train(self, resume_from=None):
        """
        Train the DQN agent.

        Args:
            resume_from: Path to model checkpoint to resume from
        """
        start_episode = 1

        if resume_from:
            self.agent.load(resume_from)

            # Extract episode number from filename if present
            # Expected format: flappybird_dqn_ep100.pth
            import re

            match = re.search(r"_ep(\d+)\.pth", resume_from)
            if match:
                start_episode = int(match.group(1)) + 1
                print(f"Resuming from episode {start_episode - 1}")
            else:
                print(f"Resuming training (episode number not found in filename)")

        print(f"\n{'='*60}")
        print(f"Starting DQN Training")
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

    parser = argparse.ArgumentParser(description="Train DQN for Flappy Bird")
    parser.add_argument(
        "--model-path",
        type=str,
        default="flappybird_dqn.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training",
    )

    args = parser.parse_args()

    trainer = DQNTrainer(
        model_path=args.model_path,
    )

    trainer.train(resume_from=args.resume)
