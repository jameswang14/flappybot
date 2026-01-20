import random
import cv2
import numpy as np
import time
import mss
import argparse
from player import FlappyBirdPlayer
from dqn_player import DQNPlayer
from cnn_dqn_player import CNNDQNPlayer
import pytesseract
import pyautogui


class FlappyBirdBot:
    def __init__(
        self,
        enable_ai=False,
        shadow_mode=False,
        tap_threshold=18,
        ground_margin=100,
        record_video=False,
        show_debug=True,
        use_dqn=False,
        dqn_model_path=None,
        use_cnn_dqn=False,
    ):
        # Initialize screen capture for QuickTime window
        self.sct = mss.mss()
        self.monitor = None
        self.bird_position = None
        self.pipes = []
        self.ground_y = None  # Y coordinate of the ground
        self.score = None  # Current score/multiplier
        self.lock_button = None  # Lock It In button position
        self.start_button = None  # Start button position
        self.game_start_time = None  # Time when game started

        # Video recording
        self.record_video = record_video
        self.video_writer = None
        self.video_frames = []
        self.frame_timestamps = []  # Track timestamps for accurate FPS

        # Debug display
        self.show_debug = show_debug

        # AI player
        self.enable_ai = enable_ai
        self.shadow_mode = shadow_mode
        self.use_dqn = use_dqn
        self.use_cnn_dqn = use_cnn_dqn
        self.bird_alive = True

        if use_cnn_dqn and (enable_ai or shadow_mode):
            # Use CNN DQN player
            self.player = CNNDQNPlayer()
            if dqn_model_path:
                self.player.load(dqn_model_path)
                print(f"Loaded CNN DQN model from {dqn_model_path}")
        elif use_dqn and (enable_ai or shadow_mode):
            # Use DQN player
            self.player = DQNPlayer()
            if dqn_model_path:
                self.player.load(dqn_model_path)
                print(f"Loaded DQN model from {dqn_model_path}")
        elif enable_ai or shadow_mode:
            # Use rule-based player
            self.player = FlappyBirdPlayer(
                tap_threshold=tap_threshold, ground_margin=ground_margin
            )
        else:
            self.player = None

        self.setup_capture()

    def reset(self):
        """Reset game state without resetting screen capture setup"""
        self.bird_position = None
        self.pipes = []
        self.ground_y = None
        self.score = None
        self.lock_button = None
        self.start_button = None
        self.game_start_time = None
        self.bird_alive = True

        # Reset video recording state
        self.video_frames = []
        self.frame_timestamps = []

        # Reset player state if it exists
        self.player = FlappyBirdPlayer(
            tap_threshold=self.player.tap_threshold,
            ground_margin=self.player.ground_margin,
        )

        print("Bot state reset")

    def setup_capture(self):
        """Setup screen capture from IPhone Mirroring window"""
        print("\n=== SETUP INSTRUCTIONS ===")
        print("1. Make sure iPhone screen is mirrored on your mac")
        print("2. Position the window left most of the monitor at full height")
        print("3. We'll capture a specific region of your screen")
        print("==========================\n")

        input("Press Enter when ready...")

        # Get monitor info
        monitors = self.sct.monitors
        monitor_num = 1  # Primary monitor
        monitor = monitors[monitor_num]

        print(f"\nMonitor resolution: {monitor['width']}x{monitor['height']}")

        # Default bounding box (you can adjust these values)
        # Format: {"top": y, "left": x, "width": w, "height": h}
        default_box = {
            "top": 65,
            "left": 5,
            "width": 300,
            "height": 650,
        }

        print(f"\nDefault capture box:")
        print(f"  Top: {default_box['top']}")
        print(f"  Left: {default_box['left']}")
        print(f"  Width: {default_box['width']}")
        print(f"  Height: {default_box['height']}")

        self.monitor = default_box

        # Take a test screenshot with the bounding box
        frame = self.get_frame()
        test_image_path = "test_capture.png"
        cv2.imwrite(test_image_path, frame)
        print(f"\nTest image saved to {test_image_path}")
        print(f"Captured resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(
            "\nPlease check the image. If it's not right, run again and adjust the box dimensions."
        )
        print(
            f"\nFinal capture box: top={self.monitor['top']}, left={self.monitor['left']}, "
            f"width={self.monitor['width']}, height={self.monitor['height']}"
        )
        input("Press any character to continue if image looks correct")

    def get_frame(self):
        """Capture a frame from the screen"""
        # Capture the screen
        screenshot = self.sct.grab(self.monitor)

        # Convert to numpy array
        frame = np.array(screenshot)

        # Convert from BGRA to BGR (remove alpha channel)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def detect_bird(self, frame):
        """Detect the bird's position in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Focus on the orange/tan body of the bird only
        # Tighter range to avoid white ground/pipes
        lower_orange = np.array([8, 80, 80])
        upper_orange = np.array([22, 255, 255])

        # Create mask for orange/tan color only
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Optional: Save mask for debugging
        cv2.imwrite("bird_mask.png", mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and aspect ratio
        valid_contours = []
        frame_height = frame.shape[0]

        for contour in contours:
            area = cv2.contourArea(contour)

            # Bird should be relatively small but not tiny
            if 100 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)

                # Bird should be roughly square-ish (not too elongated like pipes)
                aspect_ratio = w / h if h > 0 else 0

                # Bird is in the upper 80% of screen (not in the ground area)
                if 0.3 < aspect_ratio < 2.0 and y < frame_height * 0.8:
                    valid_contours.append(contour)

        if valid_contours:
            # Find the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Get center of bird
            bird_center_x = x + w // 2
            bird_center_y = y + h // 2

            self.bird_position = (bird_center_x, bird_center_y, w, h)
            return (x, y, w, h)

        return None

    def detect_score(self, frame):
        """Detect and parse the score/multiplier text using fixed coordinates"""
        frame_height, frame_width = frame.shape[:2]

        score_left = 0.36
        score_top = 0.19
        score_width = 0.30
        score_height = 0.05

        # Convert to pixel coordinates
        roi_x = int(score_left * frame_width)
        roi_y = int(score_top * frame_height)
        roi_width = int(score_width * frame_width)
        roi_height = int(score_height * frame_height)

        # Extract ROI
        roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to make text stand out
        # White text on colored background
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Use pytesseract to extract text
        # Configure for digits, decimal points, and 'x' character
        custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.x"
        try:
            text = pytesseract.image_to_string(thresh, config=custom_config)
            text = text.strip()

            if text:
                self.score = text
                return {
                    "text": text,
                    "x": roi_x,
                    "y": roi_y,
                    "width": roi_width,
                    "height": roi_height,
                }
        except Exception as e:
            # If OCR fails, just continue without score
            pass

        return None

    def detect_start_button(self, frame):
        """Detect the start button using fixed coordinates"""
        frame_height, frame_width = frame.shape[:2]

        # MANUAL ADJUSTMENT: Set these coordinates to match your start button location
        # These are relative to the frame (0-1 range), will be converted to pixels
        button_left = 0.45  # 45% from left edge
        button_top = 0.80  # 80% from top
        button_width = 0.10  # 10% of frame width
        button_height = 0.05  # 5% of frame height

        # Convert to pixel coordinates
        x = int(button_left * frame_width)
        y = int(button_top * frame_height)
        w = int(button_width * frame_width)
        h = int(button_height * frame_height)

        self.start_button = (x, y, w, h)
        return (x, y, w, h)

    def detect_lock_button(self, frame):
        """Detect the yellow 'Lock It In' button using fixed coordinates"""
        frame_height, frame_width = frame.shape[:2]

        # MANUAL ADJUSTMENT: Set these coordinates to match your button location
        # These are relative to the frame (0-1 range), will be converted to pixels
        button_left = 0.05  # 5% from left edge
        button_top = 0.80  # 70% from top (lower portion of screen)
        button_width = 0.35  # 30% of frame width
        button_height = 0.07  # 8% of frame height

        # Convert to pixel coordinates
        x = int(button_left * frame_width)
        y = int(button_top * frame_height)
        w = int(button_width * frame_width)
        h = int(button_height * frame_height)

        self.lock_button = (x, y, w, h)
        return (x, y, w, h)

    def detect_ground(self, frame):
        """Detect the ground position in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ground is typically tan/beige colored
        # Adjust these ranges based on your game's ground color
        lower_ground = np.array([10, 40, 100])
        upper_ground = np.array([30, 150, 200])

        mask = cv2.inRange(hsv, lower_ground, upper_ground)

        # Find the topmost ground pixel (scan from bottom up)
        frame_height = frame.shape[0]

        # Look in the bottom 30% of the screen
        search_start = int(frame_height * 0.7)

        for y in range(search_start, frame_height):
            row = mask[y, :]
            if (
                np.sum(row) > mask.shape[1] * 0.5 * 255
            ):  # If more than 50% of row is ground
                self.ground_y = y
                return y

        # Default to 90% down the screen if not detected
        self.ground_y = int(frame_height * 0.9)
        return self.ground_y

    def detect_pipes(self, frame):
        """Detect pipes using simple contour clustering"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green color range for pipes
        lower_green = np.array([40, 60, 80])
        upper_green = np.array([75, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Save mask for debugging
        cv2.imwrite("pipe_mask.png", mask)

        # Find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_height, frame_width = mask.shape

        # Get all green regions (filter out background bushes)
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Filter criteria:
            # 1. Minimum area to filter noise
            # 2. Maximum height to filter bushes (bushes are typically taller/larger)
            # 3. Minimum width to filter tiny artifacts
            max_height = frame_height * 0.60  # Bushes are typically taller than this

            if (
                area > 200 and h < max_height and w > 20  # Filter out large bushes
            ):  # Minimum width to filter tiny artifacts
                regions.append(
                    {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "center_x": x + w // 2,
                        "center_y": y + h // 2,
                    }
                )

        if not regions:
            self.pipes = []
            return []

        # Group regions by horizontal position (same pipe column)
        # Filter out regions that are too close to the left edge (partial pipes)
        frame_width = mask.shape[1]
        min_x_threshold = frame_width * 0.05  # Ignore leftmost 5% of screen

        regions = [r for r in regions if r["x"] > min_x_threshold]

        if not regions:
            self.pipes = []
            return []

        pipe_groups = []
        regions.sort(key=lambda r: r["center_x"])

        current_group = [regions[0]]
        for region in regions[1:]:
            # If close horizontally, add to current group
            if region["center_x"] - current_group[-1]["center_x"] < 100:
                current_group.append(region)
            else:
                pipe_groups.append(current_group)
                current_group = [region]
        pipe_groups.append(current_group)

        # Process each group into a pipe
        pipes = []
        for group in pipe_groups:
            # Sort regions by Y position to find natural gap
            sorted_regions = sorted(group, key=lambda r: r["y"])

            # Find the largest vertical gap between consecutive regions
            max_gap = 0
            gap_index = 0
            for i in range(len(sorted_regions) - 1):
                current_bottom = sorted_regions[i]["y"] + sorted_regions[i]["height"]
                next_top = sorted_regions[i + 1]["y"]
                gap = next_top - current_bottom

                if gap > max_gap:
                    max_gap = gap
                    gap_index = i

            # Split into top and bottom based on the largest gap
            # Also apply screen position filters to avoid edge cases
            if max_gap > 50:  # Minimum gap threshold
                top_regions = sorted_regions[: gap_index + 1]
                bottom_regions = sorted_regions[gap_index + 1 :]

                # Filter out bottom regions that are too low (ground level)
                bottom_regions = [
                    r for r in bottom_regions if r["center_y"] < frame_height * 0.80
                ]
            else:
                # No clear gap found, fall back to position-based classification
                top_regions = [r for r in group if r["center_y"] < frame_height * 0.45]
                bottom_regions = [
                    r
                    for r in group
                    if frame_height * 0.45 < r["center_y"] < frame_height * 0.80
                ]

            # Create unified bounding boxes
            top_pipe = None
            if top_regions:
                min_x = min(r["x"] for r in top_regions)
                min_y = min(r["y"] for r in top_regions)
                max_x = max(r["x"] + r["width"] for r in top_regions)
                max_y = max(r["y"] + r["height"] for r in top_regions)
                top_pipe = {
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                }

            bottom_pipe = None
            if bottom_regions:
                min_x = min(r["x"] for r in bottom_regions)
                min_y = min(r["y"] for r in bottom_regions)
                max_x = max(r["x"] + r["width"] for r in bottom_regions)
                max_y = max(r["y"] + r["height"] for r in bottom_regions)
                bottom_pipe = {
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                }

            # Calculate gap
            if top_pipe and bottom_pipe:
                gap_top = top_pipe["y"] + top_pipe["height"]
                gap_bottom = bottom_pipe["y"]
                gap_height = gap_bottom - gap_top
                gap_center = (gap_top + gap_bottom) // 2

                # Validate gap - must be reasonable size and position
                # Skip gaps that are too small (< 60px) or too large (> 400px)
                # Also skip gaps at very top or bottom of screen
                if (
                    gap_height < 60
                    or gap_height > 400
                    or gap_center < frame_height * 0.15
                    or gap_center > frame_height * 0.85
                ):
                    continue  # Skip this invalid gap

                # Use the most forward (rightmost) edge for the gap position
                pipe_x_right = max(
                    top_pipe["x"] + top_pipe["width"],
                    bottom_pipe["x"] + bottom_pipe["width"],
                )
                pipe_x = min(top_pipe["x"], bottom_pipe["x"])
                pipe_width = pipe_x_right - pipe_x

                pipes.append(
                    {
                        "x": pipe_x,
                        "center_x": pipe_x_right,  # Use rightmost edge as center
                        "width": pipe_width,
                        "top_pipe": top_pipe,
                        "bottom_pipe": bottom_pipe,
                        "gap_top": gap_top,
                        "gap_bottom": gap_bottom,
                        "gap_center": gap_center,
                        "gap_height": gap_height,
                    }
                )
            elif top_pipe:
                gap_top = top_pipe["y"] + top_pipe["height"]
                pipes.append(
                    {
                        "x": top_pipe["x"],
                        "center_x": top_pipe["x"]
                        + top_pipe["width"],  # Use rightmost edge
                        "width": top_pipe["width"],
                        "top_pipe": top_pipe,
                        "bottom_pipe": None,
                        "gap_top": gap_top,
                        "gap_bottom": frame_height,
                        "gap_center": gap_top + 100,
                        "gap_height": frame_height - gap_top,
                    }
                )
            elif bottom_pipe:
                gap_bottom = bottom_pipe["y"]
                pipes.append(
                    {
                        "x": bottom_pipe["x"],
                        "center_x": bottom_pipe["x"]
                        + bottom_pipe["width"],  # Use rightmost edge
                        "width": bottom_pipe["width"],
                        "top_pipe": None,
                        "bottom_pipe": bottom_pipe,
                        "gap_top": 0,
                        "gap_bottom": gap_bottom,
                        "gap_center": max(gap_bottom - 100, gap_bottom // 2),
                        "gap_height": gap_bottom,
                    }
                )

        self.pipes = pipes
        return pipes

    def process_frame(self, frame):
        """Process frame to detect game elements"""
        # Detect bird, pipes, ground, score, start button, and lock button
        bird = self.detect_bird(frame)
        pipes = self.detect_pipes(frame)
        ground_y = self.detect_ground(frame)
        # score = self.detect_score(frame)
        start_button = self.detect_start_button(frame)
        lock_button = self.detect_lock_button(frame)

        # Create a copy for visualization
        display_frame = frame.copy()

        # Draw bird detection
        if bird:
            x, y, w, h = bird
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(display_frame, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
            cv2.putText(
                display_frame,
                "Bird",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Draw pipe detections
        for i, pipe in enumerate(pipes):
            # Draw top pipe
            if pipe.get("top_pipe"):
                tp = pipe["top_pipe"]
                cv2.rectangle(
                    display_frame,
                    (tp["x"], tp["y"]),
                    (tp["x"] + tp["width"], tp["y"] + tp["height"]),
                    (0, 255, 0),
                    2,
                )

            # Draw bottom pipe
            if pipe.get("bottom_pipe"):
                bp = pipe["bottom_pipe"]
                cv2.rectangle(
                    display_frame,
                    (bp["x"], bp["y"]),
                    (bp["x"] + bp["width"], bp["y"] + bp["height"]),
                    (0, 255, 0),
                    2,
                )

            # Draw gap center line and info
            gap_center = pipe.get("gap_center", 0)
            gap_top = pipe.get("gap_top", 0)
            gap_bottom = pipe.get("gap_bottom", 0)
            x = pipe.get("x", 0)
            w = pipe.get("width", 0)

            # Draw horizontal line through gap center
            cv2.line(
                display_frame, (x, gap_center), (x + w, gap_center), (255, 0, 255), 2
            )

            # Draw gap boundaries
            cv2.line(display_frame, (x, gap_top), (x + w, gap_top), (0, 255, 255), 1)
            cv2.line(
                display_frame, (x, gap_bottom), (x + w, gap_bottom), (0, 255, 255), 1
            )

            # Label
            cv2.putText(
                display_frame,
                f"Gap {i}",
                (x, gap_center - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
            )

        # Draw ground line
        if ground_y is not None:
            frame_width = display_frame.shape[1]
            cv2.line(
                display_frame,
                (0, ground_y),
                (frame_width, ground_y),
                (0, 165, 255),  # Orange color for ground
                2,
            )
            cv2.putText(
                display_frame,
                "Ground",
                (10, ground_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
            )

        # Draw start button detection box
        if start_button is not None:
            x, y, w, h = start_button
            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 255),  # Magenta color for start button
                2,
            )
            cv2.putText(
                display_frame,
                "Start",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
            )

        # Draw lock button detection box
        if lock_button is not None:
            x, y, w, h = lock_button
            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),  # Green color for lock button
                2,
            )
            cv2.putText(
                display_frame,
                "Lock It In",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return display_frame

    def start_game(self):
        """Click to start the game using the start button bounding box"""
        # Capture a frame to get start button position
        frame = self.get_frame()
        self.detect_start_button(frame)

        if self.start_button:
            start_x_rel, start_y_rel, start_w, start_h = self.start_button
            # Click center of the start button (exact center, no randomization)
            start_x = self.monitor["left"] + start_x_rel + start_w // 2
            start_y = self.monitor["top"] + start_y_rel + start_h // 2

            print("Starting game...")
            pyautogui.click(start_x, start_y)
            time.sleep(0.1)
            pyautogui.click(start_x, start_y)

            # Start the game timer
            self.game_start_time = time.time()

    def run(self):
        """Main loop to continuously capture and process frames"""
        print("\nStarting capture loop...")
        print("Press 'esc' to quit")
        print("Press 'd' to toggle debug display")
        if self.enable_ai:
            print("AI Player: ENABLED (actually playing)")
        elif self.shadow_mode:
            print("Shadow Mode: ENABLED (showing when AI would tap)")
        else:
            print("AI Player: DISABLED (pass --ai or --shadow to enable)")

        if self.record_video:
            print("Video Recording: ENABLED (will save on exit)")
        if not self.show_debug:
            print("Debug Window: DISABLED")
        print()

        started = False

        # Track bird position history to detect game over
        bird_position_history = []
        max_history_length = 30
        position_threshold = 5  # pixels - allow small movements

        try:
            while True:
                # Capture frame
                frame = self.get_frame()

                # Process frame to detect bird and pipes
                display_frame = self.process_frame(frame)

                # Check if we should lock in (25 seconds elapsed)
                should_lock_in = False
                if self.enable_ai and self.lock_button:
                    # Check if 25 seconds have elapsed
                    if self.game_start_time is not None:
                        elapsed_time = time.time() - self.game_start_time
                        if elapsed_time >= 24:
                            print(f"\n25 seconds elapsed (time: {elapsed_time:.1f}s)")
                            should_lock_in = True

                if should_lock_in:
                    lock_x, lock_y, lock_w, lock_h = self.lock_button
                    # Calculate absolute screen coordinates
                    abs_lock_x = self.monitor["left"] + lock_x + lock_w // 2
                    abs_lock_y = self.monitor["top"] + lock_y + lock_h // 2

                    pyautogui.click(abs_lock_x, abs_lock_y)

                    cv2.putText(
                        display_frame,
                        "LOCKING IN!",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    print("Locked in after ", elapsed_time)
                    return

                # AI decision making
                if (self.enable_ai or self.shadow_mode) and self.bird_position:
                    # Get tap region (absolute screen coordinates)
                    tap_region = {
                        "x": self.monitor["left"],
                        "y": self.monitor["top"],
                        "width": self.monitor["width"],
                        "height": self.monitor["height"],
                    }

                    # Execute AI decision (with shadow mode flag and ground info)
                    if self.use_cnn_dqn:
                        # CNN DQN only needs raw frame
                        would_tap = self.player.play_step(
                            frame,
                            tap_region,
                            shadow_mode=self.shadow_mode,
                        )
                    elif self.use_dqn:
                        # DQN player needs bird position, pipes, ground
                        bird_x, bird_y, bird_w, bird_h = self.bird_position
                        bird_pos = {
                            "x": bird_x,
                            "y": bird_y,
                            "center_x": bird_x,
                            "center_y": bird_y,
                            "width": bird_w,
                            "height": bird_h,
                        }
                        would_tap = self.player.play_step(
                            bird_pos,
                            self.pipes,
                            tap_region,
                            shadow_mode=self.shadow_mode,
                            ground_y=self.ground_y,
                            frame_height=frame.shape[0],
                        )
                    else:
                        # Rule-based player also needs bird position, pipes, ground
                        bird_x, bird_y, bird_w, bird_h = self.bird_position
                        bird_pos = {
                            "x": bird_x,
                            "y": bird_y,
                            "center_x": bird_x,
                            "center_y": bird_y,
                            "width": bird_w,
                            "height": bird_h,
                        }
                        would_tap = self.player.play_step(
                            bird_pos,
                            self.pipes,
                            tap_region,
                            shadow_mode=self.shadow_mode,
                            ground_y=self.ground_y,
                            frame_height=frame.shape[0],
                        )

                    # Visual feedback for taps
                    if would_tap:
                        if self.shadow_mode:
                            # Shadow mode - show when AI would tap (yellow)
                            cv2.putText(
                                display_frame,
                                "WOULD TAP",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255),
                                2,
                            )
                        else:
                            cv2.putText(
                                display_frame,
                                "TAP",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255),
                                2,
                            )

                # Show debug window if enabled
                if self.show_debug:
                    # Resize for better viewing if needed
                    display_height = 800
                    aspect_ratio = display_frame.shape[1] / display_frame.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    display_resized = cv2.resize(
                        display_frame, (display_width, display_height)
                    )

                    # Record frame if video recording is enabled
                    if self.record_video:
                        self.video_frames.append(display_resized.copy())
                        self.frame_timestamps.append(time.time())

                    # Position window to the right of iPhone mirroring
                    window_name = "Flappy Bird Bot - Debug"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.moveWindow(window_name, 800, 0)  # Move to x=800, y=0
                    cv2.imshow(window_name, display_resized)

                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("d"):
                        self.show_debug = not self.show_debug
                        if not self.show_debug:
                            cv2.destroyAllWindows()
                    if not started:
                        # put window into focus
                        pyautogui.click(100, 100)
                else:
                    # Small delay when not showing debug
                    time.sleep(0.001)

                # Print detection info
                if self.bird_position:
                    bird_x, bird_y, _, _ = self.bird_position
                    if self.enable_ai:
                        status = "AI Playing"
                    elif self.shadow_mode:
                        status = "Shadow Mode"
                    else:
                        status = "Watching"
                    print(
                        f"\r{status} | Bird: ({bird_x}, {bird_y}) | Pipes: {len(self.pipes)}",
                        end="",
                    )

                    # Track bird position to detect game over
                    if started and self.enable_ai:
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
                                    "\n\nBird position hasn't changed in 20 frames - game over detected!"
                                )
                                print(
                                    "\n\nTime Elapsed",
                                    time.time() - self.game_start_time,
                                )
                                self.bird_alive = False
                                return

                if not started and self.enable_ai:
                    self.start_game()
                    started = True

        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            # Save video if recording was enabled
            if not self.use_dqn:
                if self.record_video and self.video_frames:
                    self._save_video()

                cv2.destroyAllWindows()
                print("Capture stopped")

    def _save_video(self):
        """Save recorded frames to a video file"""
        if not self.video_frames:
            print("No frames to save")
            return

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"flappybird_recording_{timestamp}.mp4"

        print(f"\nSaving video with {len(self.video_frames)} frames...")

        # Calculate actual FPS from timestamps
        if len(self.frame_timestamps) > 1:
            total_duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
            actual_fps = len(self.frame_timestamps) / total_duration
            print(f"Actual capture rate: {actual_fps:.2f} fps")
            # Use actual FPS for video, capped at 60
            fps = min(actual_fps, 60)
        else:
            fps = 30  # Fallback

        # Get frame dimensions
        height, width = self.video_frames[0].shape[:2]

        # Create video writer (using mp4v codec for MP4 format)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Write all frames
        for frame in self.video_frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved to: {output_file} at {fps:.2f} fps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flappy Bird Bot")
    parser.add_argument(
        "--ai", action="store_true", help="Enable AI to play the game automatically"
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Shadow mode: show when AI would tap without actually tapping",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record debug screen to video file (saved on exit)",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug window display",
    )
    parser.add_argument(
        "--dqn",
        action="store_true",
        help="Use DQN (Deep Q-Network) instead of rule-based AI",
    )
    parser.add_argument(
        "--cnn-dqn",
        action="store_true",
        help="Use CNN DQN (learns from raw pixels) instead of rule-based AI",
    )
    parser.add_argument(
        "--dqn-model",
        type=str,
        default=None,
        help="Path to DQN model file to load",
    )
    parser.add_argument("--num-games", type=int, default=1)

    args = parser.parse_args()

    bot = FlappyBirdBot(
        enable_ai=args.ai,
        shadow_mode=args.shadow,
        record_video=args.record,
        show_debug=not args.no_debug,
        use_dqn=args.dqn,
        use_cnn_dqn=args.cnn_dqn,
        dqn_model_path=args.dqn_model,
    )
    # put window into focus
    pyautogui.click(100, 100)
    time.sleep(1)
    for x in range(args.num_games):
        if x > 0:
            bot.reset()
            time.sleep(random.randint(10, 15))

        bot.run()
