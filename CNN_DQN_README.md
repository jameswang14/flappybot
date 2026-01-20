# CNN DQN (Convolutional Deep Q-Network) for Flappy Bird

This implementation adds a CNN-based DQN agent that learns to play Flappy Bird directly from raw pixels, without hand-crafted features.

## Differences from Regular DQN

| Feature | Regular DQN | CNN DQN |
|---------|-------------|---------|
| **Input** | 7 hand-crafted features (bird position, pipe distances, etc.) | Raw pixels (84x84 grayscale frame) |
| **Feature extraction** | Manual (bird detection, pipe detection) | Learned via CNN layers |
| **State size** | Small (7 numbers) | Large (84x84 = 7,056 pixels) |
| **Network architecture** | Fully connected layers | CNN + fully connected layers |
| **Training speed** | Faster (smaller input) | Slower (larger input, more parameters) |
| **Potential** | Limited by feature engineering | Can learn complex visual patterns |

## How CNN DQN Works

The CNN DQN learns by:
1. **Observing raw game frames** (converted to 84x84 grayscale)
2. **Processing through convolutional layers** to extract spatial features
3. **Combining with time since last tap** feature
4. **Taking actions** (tap or don't tap)
5. **Receiving rewards** and learning which visual patterns lead to success

### State Features (2 features):
- **Raw game frame** (84x84 grayscale image, processed through CNN)
- **Time since last tap** (normalized, capped at 1 second)

### Network Architecture:

```
Raw Frame (84x84 grayscale)
    ↓
Conv2d(32 filters, 8x8 kernel, stride 4) + ReLU
    ↓
Conv2d(64 filters, 4x4 kernel, stride 2) + ReLU
    ↓
Conv2d(64 filters, 3x3 kernel, stride 1) + ReLU
    ↓
Flatten (3136 features)
    ↓
Concatenate with time_since_tap (+1 feature)
    ↓
Dense Layer (512 neurons) + ReLU
    ↓
Output (2 Q-values: one for each action)
```

### Actions:
- 0: Don't tap (let bird fall)
- 1: Tap (make bird jump)

### Rewards:
- +0.1: Small reward for staying alive each frame
- -10.0: Large penalty for crashing

## Training

Train a new CNN DQN model:

```bash
python train_cnn_dqn.py --episodes 1000
```

### Training Options:

```bash
python train_cnn_dqn.py \
  --episodes 1000 \              # Number of training episodes
  --model-path my_cnn_model.pth  # Where to save the model
```

### Resume Training:

```bash
python train_cnn_dqn.py --resume flappybird_cnn_dqn_ep100.pth --episodes 500
```

## Using a Trained Model

### Test the trained model in shadow mode:

```bash
python vision.py --shadow --cnn-dqn --dqn-model flappybird_cnn_dqn_best.pth
```

### Let the CNN DQN play the game:

```bash
python vision.py --ai --cnn-dqn --dqn-model flappybird_cnn_dqn_best.pth
```

### Record the CNN DQN playing:

```bash
python vision.py --ai --cnn-dqn --dqn-model flappybird_cnn_dqn_best.pth --record
```

## Hyperparameters

Default hyperparameters in `cnn_dqn_player.py`:

- **Learning rate**: 0.00025 (lower than regular DQN due to larger network)
- **Gamma (discount factor)**: 0.99
- **Epsilon decay**: 0.995
- **Replay buffer size**: 10,000 transitions
- **Batch size**: 32 (smaller than regular DQN for memory efficiency)
- **Target network update frequency**: Every 1000 steps (more frequent updates)

## Training Tips

1. **CNN DQN requires more episodes**: Expect to train for 500-1000+ episodes before seeing good performance, as it needs to learn visual features from scratch.

2. **Monitor GPU usage**: CNN training benefits greatly from GPU acceleration. Check that PyTorch is using your GPU:
   ```
   CNN DQN using device: cuda  # Good!
   CNN DQN using device: cpu   # Will be slower
   ```

3. **Frame preprocessing is important**: Frames are converted to 84x84 grayscale and normalized to [0, 1] range.

4. **Best model is auto-saved**: The model with the highest reward is automatically saved as `*_best.pth`

5. **Compare all three approaches**:
   ```bash
   python vision.py --ai                                    # Rule-based AI
   python vision.py --ai --dqn --dqn-model model.pth        # Feature-based DQN
   python vision.py --ai --cnn-dqn --dqn-model cnn_model.pth  # CNN DQN
   ```

## Advantages of CNN DQN

1. **No feature engineering required**: Doesn't need bird detection, pipe detection, or other hand-crafted features
2. **Can learn complex patterns**: May discover visual patterns humans didn't think of
3. **More general approach**: Same architecture could work for other visual games
4. **End-to-end learning**: Learns everything from pixels to actions

## Disadvantages of CNN DQN

1. **Slower training**: Requires more computation and episodes to converge
2. **More memory usage**: Stores images in replay buffer instead of small feature vectors
3. **Harder to debug**: Can't easily see what features the network is using
4. **May need more tuning**: Hyperparameters may need adjustment for best performance

## Troubleshooting

**Issue**: Training is very slow
- Make sure PyTorch is using GPU (check startup message)
- Reduce batch size or buffer size
- Close debug window during training (`show_debug=False` is default)

**Issue**: Agent doesn't improve
- Try training for more episodes (CNN needs more data)
- Adjust learning rate (try 0.0001 or 0.0005)
- Check that frames are being preprocessed correctly

**Issue**: Out of memory errors
- Reduce batch size (try 16 or 8)
- Reduce replay buffer size (try 5000)
- Ensure you're not accumulating frames in memory

## Files

- `cnn_dqn_player.py`: CNN DQN neural network and agent implementation
- `train_cnn_dqn.py`: Training script for CNN DQN
- `vision.py`: Updated to support CNN DQN player
- `CNN_DQN_README.md`: This file
