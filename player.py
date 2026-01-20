import pyautogui
import time
import random


class FlappyBirdPlayer:
    def __init__(self, tap_threshold=30, lookahead_distance=250, ground_margin=80):
        """
        Simple rule-based AI for Flappy Bird.

        Args:
            tap_threshold: How many pixels below the gap center before tapping
            lookahead_distance: How far ahead (in pixels) to look for the next pipe
            ground_margin: How many pixels above ground to maintain as safety margin
        """
        self.tap_threshold = tap_threshold
        self.lookahead_distance = lookahead_distance
        self.ground_margin = ground_margin
        self.last_tap_time = 0
        self.min_tap_interval = 0.15  # Minimum time between taps (seconds)

        # Adaptive threshold parameters
        self.base_tap_threshold = tap_threshold
        self.strict_threshold_multiplier = 2
        self.threshold_decay_time = (
            0.3  # Time for strictness to fully wear off (seconds)
        )

        # Proactive tap parameters (when bird hasn't tapped in a while)
        self.proactive_threshold_multiplier = 0.9
        self.proactive_tap_delay = (
            0.6  # Time without tap before becoming proactive (seconds) (was 0.8)
        )

        # Track game start time for initial centering behavior
        self.game_start_time = None

    def get_current_tap_threshold(self):
        """
        Calculate the current tap threshold based on time since last tap.
        - Stricter immediately after tapping (prevents rapid fire)
        - Less strict if no tap for a while (encourages proactive tapping)

        Returns:
            float: Current tap threshold
        """
        current_time = time.time()
        time_since_tap = current_time - self.last_tap_time

        # Phase 1: Strict period right after tap (0 to threshold_decay_time)
        if time_since_tap < self.threshold_decay_time:
            # Calculate decay factor (1.0 = most strict, 0.0 = base threshold)
            decay_factor = 1.0 - (time_since_tap / self.threshold_decay_time)

            # Interpolate between base and strict threshold
            strict_threshold = (
                self.base_tap_threshold * self.strict_threshold_multiplier
            )
            current_threshold = (
                self.base_tap_threshold
                + (strict_threshold - self.base_tap_threshold) * decay_factor
            )
            return current_threshold

        # Phase 2: Normal period (threshold_decay_time to proactive_tap_delay)
        if time_since_tap < self.proactive_tap_delay:
            return self.base_tap_threshold

        # Phase 3: Proactive period (after proactive_tap_delay)
        # Gradually decrease threshold to encourage tapping
        time_in_proactive = time_since_tap - self.proactive_tap_delay
        max_proactive_duration = 0.4  # Max time to reach full proactivity

        if time_in_proactive >= max_proactive_duration:
            # Fully proactive
            proactive_threshold = (
                self.base_tap_threshold * self.proactive_threshold_multiplier
            )
            return proactive_threshold

        # Gradually interpolate from base to proactive threshold
        proactive_factor = time_in_proactive / max_proactive_duration
        proactive_threshold = (
            self.base_tap_threshold * self.proactive_threshold_multiplier
        )
        current_threshold = (
            self.base_tap_threshold
            - (self.base_tap_threshold - proactive_threshold) * proactive_factor
        )

        return current_threshold

    def decide_action(self, bird_pos, pipes, ground_y=None, frame_height=None):
        """
        Decide whether to tap based on bird position and pipes.

        Args:
            bird_pos: Dict with bird position {'x': int, 'y': int, 'center_y': int}
            pipes: List of pipe dicts with gap information
            ground_y: Y coordinate of the ground (optional)
            frame_height: Height of the frame (optional, for center calculation)

        Returns:
            bool: True if should tap, False otherwise
        """
        if not bird_pos:
            return False

        # Get bird's vertical center
        bird_y = bird_pos.get("center_y", bird_pos.get("y", 0))
        bird_x = bird_pos.get("center_x", bird_pos.get("x", 0))

        # PRIORITY 1: Avoid crashing into ground
        if ground_y is not None:
            distance_to_ground = ground_y - bird_y
            if distance_to_ground < self.ground_margin:
                return True  # Emergency tap to avoid ground

        # PRIORITY 2: Stay centered for first 2 seconds of game
        if self.game_start_time is None:
            self.game_start_time = time.time()

        time_since_start = time.time() - self.game_start_time

        # For the first 2 seconds, keep bird centered
        if time_since_start < 2.0:
            if frame_height:
                screen_middle = frame_height // 2
                # Tap if bird is below middle (with 30px buffer above center)
                should_tap = bird_y > screen_middle - 30
                print(
                    f"\n[CENTERING] time={time_since_start:.1f}s bird_y={bird_y}, middle={screen_middle}, tapping={should_tap}"
                )
                return should_tap
            else:
                # No frame height info, just avoid ground
                return False

        # After 2 seconds, navigate through pipes normally
        if not pipes:
            # No pipes detected, just maintain altitude
            return False

        # Find the next pipe ahead of the bird
        next_pipe = self._find_next_pipe(bird_x, pipes)

        if not next_pipe:
            # No upcoming pipes, just maintain altitude without crashing
            return False

        # Get the gap center Y position
        gap_center = next_pipe.get("gap_center", 0)

        # Calculate how far below the gap center the bird is
        distance_below_gap = bird_y - gap_center

        # Get current adaptive threshold (stricter right after tapping)
        current_threshold = self.get_current_tap_threshold()

        # Tap if bird is too far below the gap center
        should_tap = distance_below_gap > current_threshold

        return should_tap

    def _find_next_pipe(self, bird_x, pipes):
        """
        Find the next pipe that the bird needs to navigate through.

        Args:
            bird_x: Bird's horizontal center position
            pipes: List of pipe dicts

        Returns:
            Dict: The next pipe ahead of the bird, or None
        """
        # Filter pipes that are ahead of or overlapping with the bird
        upcoming_pipes = [p for p in pipes if p["x"] + p["width"] > bird_x - 13]

        if not upcoming_pipes:
            return None

        # Sort by horizontal distance and return the CLOSEST (not farthest)
        upcoming_pipes.sort(key=lambda p: p["x"])
        return upcoming_pipes[-1]

    def execute_tap(self, tap_region, randomize=True):
        """
        Execute a screen tap within the specified region.

        Args:
            tap_region: Dict with screen coordinates {'x': int, 'y': int, 'width': int, 'height': int}
            randomize: Whether to add random offset to tap position (default: True)
        """
        current_time = time.time()

        # Prevent tapping too frequently
        if current_time - self.last_tap_time < self.min_tap_interval:
            return

        # Calculate tap position
        center_x = tap_region["x"] + tap_region["width"] // 2
        center_y = tap_region["y"] + tap_region["height"] // 2

        if randomize:
            # Add random offset within ±20% of the region dimensions for gameplay
            offset_x = random.randint(
                -tap_region["width"] // 5, tap_region["width"] // 5
            )
            offset_y = random.randint(
                -tap_region["height"] // 5, tap_region["height"] // 5
            )
            tap_x = center_x + offset_x
            tap_y = center_y + offset_y
        else:
            # Use exact center for UI buttons
            tap_x = center_x
            tap_y = center_y

        # Execute the tap
        pyautogui.click(tap_x, tap_y)
        self.last_tap_time = current_time

    def play_step(
        self,
        bird_pos,
        pipes,
        tap_region,
        shadow_mode=False,
        ground_y=None,
        frame_height=None,
    ):
        """
        Execute one step of gameplay: decide and act.

        Args:
            bird_pos: Dict with bird position
            pipes: List of pipe dicts
            tap_region: Dict with screen coordinates for tapping
            shadow_mode: If True, only decide but don't actually tap
            ground_y: Y coordinate of the ground (optional)
            frame_height: Height of the frame (optional)

        Returns:
            bool: True if would tap (or did tap), False otherwise
        """
        should_tap = self.decide_action(
            bird_pos, pipes, ground_y=ground_y, frame_height=frame_height
        )

        if should_tap and not shadow_mode:
            self.execute_tap(tap_region)

        return should_tap
