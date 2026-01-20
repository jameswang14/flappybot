# DQN (Deep Q-Network) for Flappy Bird

This implementation adds a Deep Q-Network agent that learns to play Flappy Bird through reinforcement learning.

## Installation

Install PyTorch and dependencies:

```bash
pip install -r requirements_dqn.txt
```

## How DQN Works

The DQN learns by:
1. **Observing the game state** (bird position, pipe locations, gap centers, etc.)
2. **Taking actions** (tap or don't tap)
3. **Receiving rewards** (positive for surviving and passing pipes, negative for crashing)
4. **Learning** which actions lead to better outcomes over time

### State Features (7 features):
- Bird Y position (normalized)
- Vertical distance to gap center
- Horizontal distance to next pipe
- Gap size
- Distance to ground
- Pipe opening size
- Time since last tap (normalized, capped at 1 second)

### Actions:
- 0: Don't tap (let bird fall)
- 1: Tap (make bird jump)

### Rewards:
- +0.1: Small reward for staying alive each frame
- +0.2 to +0.5: Reward for staying near gap center
- +5.0: Large reward for passing a pipe
- -0.5: Penalty for being too close to ground or too high
- -10.0: Large penalty for crashing

## Training

Train a new DQN model:

```bash
python train_dqn.py --episodes 500
```

### Training Options:

```bash
python train_dqn.py \
  --episodes 500 \           # Number of training episodes
  --max-steps 3000 \         # Max steps per episode
  --save-interval 50 \       # Save model every N episodes
  --model-path my_model.pth  # Where to save the model
```

### Resume Training:

```bash
python train_dqn.py --resume flappybird_dqn_ep100.pth --episodes 200
```

## Using a Trained Model

### Test the trained model in shadow mode (shows decisions without playing):

```bash
python vision.py --shadow --dqn --dqn-model flappybird_dqn_best.pth
```

### Let the DQN play the game:

```bash
python vision.py --ai --dqn --dqn-model flappybird_dqn_best.pth
```

### Record the DQN playing:

```bash
python vision.py --ai --dqn --dqn-model flappybird_dqn_best.pth --record
```

## Training Tips

1. **Start with exploration**: The agent begins with high epsilon (1.0) which means it explores randomly. Over time, epsilon decreases and it relies more on learned knowledge.

2. **Monitor progress**: Training will print stats every 10 episodes:
   - Average reward (higher is better)
   - Average episode length (longer means it survived longer)
   - Average loss (should decrease over time)
   - Epsilon (exploration rate, decreases over time)

3. **Best model is auto-saved**: The model with the highest reward is automatically saved as `*_best.pth`

4. **Training takes time**: Expect to run 300-500 episodes before seeing good performance. Each episode takes 30-60 seconds.

5. **Compare with rule-based**: You can still use the rule-based AI:
   ```bash
   python vision.py --ai  # Rule-based AI
   python vision.py --ai --dqn --dqn-model model.pth  # DQN AI
   ```

## Model Architecture

```
Input (6 features)
    ↓
Dense Layer (128 neurons) + ReLU
    ↓
Dense Layer (128 neurons) + ReLU
    ↓
Dense Layer (128 neurons) + ReLU
    ↓
Output (2 Q-values: one for each action)
```

## Hyperparameters

Default hyperparameters in `dqn_player.py`:

- **Learning rate**: 0.0005
- **Gamma (discount factor)**: 0.99
- **Epsilon decay**: 0.995
- **Replay buffer size**: 10,000 transitions
- **Batch size**: 64
- **Target network update frequency**: Every 100 steps

You can modify these in the `DQNPlayer` initialization.

## Troubleshooting

**Issue**: Training is unstable or agent doesn't improve
- Try reducing learning rate (e.g., 0.0001)
- Increase replay buffer size
- Adjust reward function in `train_dqn.py`

**Issue**: Agent learns but then forgets
- This is "catastrophic forgetting" - increase target network update frequency
- Save checkpoints more frequently

**Issue**: Training is too slow
- Reduce `--max-steps` per episode
- Train without debug window (already default in training script)
- Use GPU if available (automatically detected)

## Expected Results

After 300-500 episodes of training:
- Rule-based AI typically survives 20-30 seconds
- DQN can potentially surpass this with enough training
- Best agents can survive the full 30 seconds consistently

## Files

- `dqn_player.py`: DQN neural network and agent implementation
- `train_dqn.py`: Training script with episode management
- `vision.py`: Updated to support both rule-based and DQN players
- `requirements_dqn.txt`: PyTorch dependencies
