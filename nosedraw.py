import cv2
import numpy as np
import mediapipe as mp
import random
import os
from skimage.metrics import structural_similarity as ssim
import time
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NoseDrawingGame:
    def __init__(self):
        print("Initializing NoseDrawingGame...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("Face mesh initialized")

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

        self.objects = ["window", "star", "triangle", "cloud", "tree", "house", "flower"]
        self.current_object = random.choice(self.objects)
        self.reference_image = self.generate_reference_image(self.current_object)

        self.nose_tip_idx = 4
        self.prev_nose_pos = None
        self.nose_history = []
        self.smoothing_window = 10
        self.min_drawing_distance = 1

        self.game_state = "menu"  # 'menu' > 'playing' > 'paused' > 'done'
        self.similarity_score = 0
        self.start_time = None
        self.duration = 60  # Increased to 60 seconds as per query
        self.pause_start_time = None
        self.total_pause_time = 0
        
        # Snapshot functionality
        self.snapshots = []
        self.snapshot_times = []
        
        # Window management
        self.canvas_window_created = False
        
        # High score system
        self.high_scores_file = "high_scores.json"
        self.high_scores = self.load_high_scores()
        self.current_player_name = ""
        
        # Game flow states
        self.name_entered = False
        
        print("Game initialized successfully")

    def generate_reference_image(self, obj_type):
        ref = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        center = (self.canvas_width // 2, self.canvas_height // 2)
        size = 100

        if obj_type == "window":
            # Draw a window with frame and panes
            window_left = center[0] - size
            window_right = center[0] + size
            window_top = center[1] - size
            window_bottom = center[1] + size
            
            # Outer window frame
            cv2.rectangle(ref, (window_left, window_top), (window_right, window_bottom), (0, 0, 0), 4)
            
            # Inner frame (thinner)
            inner_left = window_left + 20
            inner_right = window_right - 20
            inner_top = window_top + 20
            inner_bottom = window_bottom - 20
            cv2.rectangle(ref, (inner_left, inner_top), (inner_right, inner_bottom), (0, 0, 0), 2)
            
            # Vertical divider
            cv2.line(ref, (center[0], inner_top), (center[0], inner_bottom), (0, 0, 0), 2)
            
            # Horizontal divider
            cv2.line(ref, (inner_left, center[1]), (inner_right, center[1]), (0, 0, 0), 2)
            
            # Window sill (bottom ledge)
            cv2.rectangle(ref, (window_left - 10, window_bottom), (window_right + 10, window_bottom + 15), (0, 0, 0), 2)
        elif obj_type == "star":
            points = []
            for i in range(5):
                outer = (
                    int(center[0] + size * np.cos(2 * np.pi * i / 5)),
                    int(center[1] + size * np.sin(2 * np.pi * i / 5))
                )
                inner = (
                    int(center[0] + (size // 2) * np.cos(2 * np.pi * i / 5 + np.pi / 5)),
                    int(center[1] + (size // 2) * np.sin(2 * np.pi * i / 5 + np.pi / 5))
                )
                points.append(outer)
                points.append(inner)
            for i in range(len(points)):
                cv2.line(ref, points[i], points[(i + 1) % len(points)], (0, 0, 0), 2)
        elif obj_type == "triangle":
            points = np.array([
                [center[0], center[1] - size],
                [center[0] - size, center[1] + size],
                [center[0] + size, center[1] + size]
            ], np.int32)
            cv2.polylines(ref, [points], True, (0, 0, 0), 3)
        elif obj_type == "cloud":
            # Draw a cloud with multiple overlapping circles
            cloud_centers = [
                (center[0] - 40, center[1] - 20),
                (center[0] + 20, center[1] - 30),
                (center[0] + 50, center[1] - 10),
                (center[0] + 30, center[1] + 20),
                (center[0] - 20, center[1] + 30),
                (center[0] - 50, center[1] + 10)
            ]
            for cloud_center in cloud_centers:
                cv2.circle(ref, cloud_center, 35, (0, 0, 0), 2)
        elif obj_type == "tree":
            # Draw a tree with trunk and foliage
            # Trunk
            trunk_bottom = (center[0], center[1] + size)
            trunk_top = (center[0], center[1] - size//2)
            cv2.line(ref, trunk_bottom, trunk_top, (0, 0, 0), 8)
            
            # Foliage (multiple circles for tree crown)
            foliage_centers = [
                (center[0] - 30, center[1] - size//2),
                (center[0] + 30, center[1] - size//2),
                (center[0], center[1] - size),
                (center[0] - 20, center[1] - size + 20),
                (center[0] + 20, center[1] - size + 20)
            ]
            for foliage_center in foliage_centers:
                cv2.circle(ref, foliage_center, 25, (0, 0, 0), 2)
        elif obj_type == "house":
            # Draw a house with roof, walls, door, and windows
            # Main house rectangle
            house_left = center[0] - size
            house_right = center[0] + size
            house_top = center[1] - size//2
            house_bottom = center[1] + size
            cv2.rectangle(ref, (house_left, house_top), (house_right, house_bottom), (0, 0, 0), 3)
            
            # Roof (triangle)
            roof_points = np.array([
                [house_left - 20, house_top],
                [center[0], house_top - 40],
                [house_right + 20, house_top]
            ], np.int32)
            cv2.polylines(ref, [roof_points], True, (0, 0, 0), 3)
            
            # Door
            door_width = 30
            door_height = 50
            door_left = center[0] - door_width//2
            door_top = house_bottom - door_height
            cv2.rectangle(ref, (door_left, door_top), (door_left + door_width, house_bottom), (0, 0, 0), 2)
            
            # Windows
            window_size = 25
            # Left window
            cv2.rectangle(ref, (house_left + 20, house_top + 30), 
                         (house_left + 20 + window_size, house_top + 30 + window_size), (0, 0, 0), 2)
            # Right window
            cv2.rectangle(ref, (house_right - 20 - window_size, house_top + 30), 
                         (house_right - 20, house_top + 30 + window_size), (0, 0, 0), 2)
        elif obj_type == "flower":
            # Draw a flower with petals, stem, and leaves
            # Stem
            stem_bottom = (center[0], center[1] + size)
            stem_top = (center[0], center[1] - size//3)
            cv2.line(ref, stem_bottom, stem_top, (0, 0, 0), 4)
            
            # Petals (multiple circles around center)
            petal_centers = []
            for i in range(8):
                angle = 2 * np.pi * i / 8
                petal_x = int(center[0] + 30 * np.cos(angle))
                petal_y = int(center[1] - size//3 + 30 * np.sin(angle))
                petal_centers.append((petal_x, petal_y))
            
            for petal_center in petal_centers:
                cv2.circle(ref, petal_center, 15, (0, 0, 0), 2)
            
            # Center of flower
            cv2.circle(ref, (center[0], center[1] - size//3), 12, (0, 0, 0), 2)
            
            # Leaves
            leaf_positions = [
                (center[0] - 20, center[1] + size//3),
                (center[0] + 20, center[1] + size//2)
            ]
            for leaf_pos in leaf_positions:
                cv2.ellipse(ref, leaf_pos, (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        return ref

    def smooth_nose_position(self, current_pos):
        self.nose_history.append(current_pos)
        if len(self.nose_history) > self.smoothing_window:
            self.nose_history.pop(0)
        avg_x = int(np.mean([pt[0] for pt in self.nose_history]))
        avg_y = int(np.mean([pt[1] for pt in self.nose_history]))
        return (avg_x, avg_y)

    def draw_smooth_line(self, start_pos, end_pos):
        if start_pos is None or end_pos is None:
            return
        # Draw regardless of distance
        cv2.line(self.canvas, start_pos, end_pos, (0, 0, 0), 3)

    def get_nose_position(self, landmarks, img_width, img_height):
        nose = landmarks.landmark[self.nose_tip_idx]
        return int(nose.x * img_width), int(nose.y * img_height)

    def calculate_similarity(self):
        # Convert to grayscale for comparison
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM score
        ssim_score, _ = ssim(gray_canvas, gray_ref, full=True)
        
        # Additional penalties for tougher scoring
        # Count black pixels in reference vs canvas
        ref_black_pixels = np.sum(gray_ref < 128)
        canvas_black_pixels = np.sum(gray_canvas < 128)
        
        # Penalty for too much or too little drawing
        if ref_black_pixels > 0:
            pixel_ratio = canvas_black_pixels / ref_black_pixels
            if pixel_ratio < 0.3 or pixel_ratio > 2.0:  # Too little or too much drawing
                ssim_score *= 0.5
        
        # Edge detection comparison for shape accuracy
        ref_edges = cv2.Canny(gray_ref, 50, 150)
        canvas_edges = cv2.Canny(gray_canvas, 50, 150)
        
        # Compare edge similarity
        edge_similarity = np.sum(cv2.bitwise_and(ref_edges, canvas_edges)) / (np.sum(ref_edges) + 1)
        ssim_score *= (0.7 + 0.3 * edge_similarity)
        
        # Final score with much stricter scaling
        final_score = round(ssim_score * 100, 2)
        
        # Apply additional penalties for common mistakes
        if final_score > 80:
            # Check for basic shape recognition
            if self.current_object in ["window", "triangle"]:
                # For basic shapes, be even stricter
                final_score *= 0.8
            elif self.current_object in ["cloud", "tree", "house", "flower"]:
                # For complex shapes, check for key elements
                if self.current_object == "cloud" and canvas_black_pixels < ref_black_pixels * 0.5:
                    final_score *= 0.6
                elif self.current_object == "tree" and canvas_black_pixels < ref_black_pixels * 0.4:
                    final_score *= 0.5
                elif self.current_object == "house" and canvas_black_pixels < ref_black_pixels * 0.6:
                    final_score *= 0.7
                elif self.current_object == "flower" and canvas_black_pixels < ref_black_pixels * 0.5:
                    final_score *= 0.6
        
        return max(0, min(100, final_score))

    def load_high_scores(self):
        """Load high scores from file"""
        try:
            if os.path.exists(self.high_scores_file):
                with open(self.high_scores_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading high scores: {e}")
        return []

    def save_high_scores(self):
        """Save high scores to file"""
        try:
            with open(self.high_scores_file, 'w') as f:
                json.dump(self.high_scores, f, indent=2)
        except Exception as e:
            print(f"Error saving high scores: {e}")

    def add_high_score(self, player_name, score, shape):
        """Add a new high score"""
        new_score = {
            "name": player_name,
            "score": score,
            "shape": shape,
            "date": time.strftime("%Y-%m-%d %H:%M")
        }
        
        # Check if this exact score already exists
        for existing_score in self.high_scores:
            if (existing_score["name"] == player_name and 
                existing_score["score"] == score and 
                existing_score["shape"] == shape):
                return  # Don't add duplicate
        
        self.high_scores.append(new_score)
        # Sort by score (highest first)
        self.high_scores.sort(key=lambda x: x["score"], reverse=True)
        # Keep only top 10 scores
        self.high_scores = self.high_scores[:10]
        self.save_high_scores()

    def get_top_scores(self, count=5):
        """Get top N scores"""
        return self.high_scores[:count]

    def show_name_input_dialog(self):
        """Show dialog to input player name"""
        player_name = ""
        name_window = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        while True:
            # Clear and redraw window
            name_window[:] = 255
            cv2.putText(name_window, "Enter Your Name:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(name_window, f"Shape to draw: {self.current_object}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            cv2.putText(name_window, "Type your name:", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(name_window, f"[{player_name}]", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(name_window, "Press ENTER to continue, ESC for Anonymous", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            
            cv2.imshow("Enter Name", name_window)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key
                cv2.destroyWindow("Enter Name")
                return "Anonymous"
            elif key == 13:  # Enter key
                cv2.destroyWindow("Enter Name")
                return player_name if player_name else "Anonymous"
            elif key == 8:  # Backspace
                player_name = player_name[:-1]
            elif 32 <= key <= 126:  # Printable ASCII characters
                if len(player_name) < 15:  # Limit name length
                    player_name += chr(key)
        
        cv2.destroyWindow("Enter Name")
        return "Anonymous"

    def take_snapshot(self):
        """Take a snapshot of the current canvas"""
        snapshot = self.canvas.copy()
        self.snapshots.append(snapshot)
        self.snapshot_times.append(time.time() - self.start_time - self.total_pause_time)

    def show_snapshot_menu(self):
        """Display snapshot menu"""
        if not self.snapshots:
            return "menu"
        
        snapshot_window = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(snapshot_window, "Snapshots Taken:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        for i, (snapshot, timestamp) in enumerate(zip(self.snapshots, self.snapshot_times)):
            y_pos = 100 + i * 120
            cv2.putText(snapshot_window, f"Snapshot {i+1} at {timestamp:.1f}s", (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Resize snapshot for display
            small_snapshot = cv2.resize(snapshot, (200, 150))
            snapshot_window[y_pos+10:y_pos+160, 50:250] = small_snapshot
        
        cv2.putText(snapshot_window, "Press any key to return", (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Snapshots", snapshot_window)
        
        cv2.waitKey(0)
        cv2.destroyWindow("Snapshots")
        return "menu"

    def pause_game(self):
        """Pause the game"""
        if self.game_state == "playing":
            self.game_state = "paused"
            self.pause_start_time = time.time()

    def resume_game(self):
        """Resume the game"""
        if self.game_state == "paused":
            self.game_state = "playing"
            if self.pause_start_time:
                self.total_pause_time += time.time() - self.pause_start_time
                self.pause_start_time = None

    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas[:] = 255
        self.prev_nose_pos = None
        self.nose_history.clear()

    def show_game_menu(self):
        """Display the main game menu"""
        menu_window = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        cv2.putText(menu_window, "Nose Draw Game", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 255), 4)
        cv2.putText(menu_window, f"Draw a {self.current_object}", (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Left side - Instructions (narrower to prevent overflow)
        cv2.putText(menu_window, "GAME INSTRUCTIONS:", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(menu_window, "1. Position your face in front of the camera", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(menu_window, "2. Move your nose to draw on the canvas", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(menu_window, "3. Try to match the reference shape", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(menu_window, "4. Press ENTER when you're done drawing", (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(menu_window, "CONTROLS:", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(menu_window, "ENTER - Start Game / Submit Drawing", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(menu_window, "P - Pause/Resume Game", (50, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(menu_window, "C - Clear Canvas", (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(menu_window, "S - Take Snapshot", (50, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(menu_window, "V - View Snapshots", (50, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(menu_window, "Q - Return to Menu", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(menu_window, "ESC - Quit Game", (50, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add reference image of current object
        cv2.putText(menu_window, "REFERENCE SHAPE:", (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        # Create a small reference image
        ref_img = self.generate_reference_image(self.current_object)
        # Resize to small size for menu
        small_ref = cv2.resize(ref_img, (150, 100))
        # Place it below the instructions
        menu_window[700:800, 50:200] = small_ref
        
        # Right side - High Scores
        cv2.putText(menu_window, "TOP 5 HIGH SCORES:", (700, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        top_scores = self.get_top_scores(5)
        if top_scores:
            for i, score_data in enumerate(top_scores):
                y_pos = 220 + i * 30
                score_text = f"{i+1}. {score_data['name']} - {score_data['score']}%"
                cv2.putText(menu_window, score_text, (700, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                # Show shape on next line
                shape_text = f"   Shape: {score_data['shape']}"
                cv2.putText(menu_window, shape_text, (700, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        else:
            cv2.putText(menu_window, "No scores yet!", (700, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            cv2.putText(menu_window, "Be the first!", (700, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        
        # Add separator line between instructions and scores
        cv2.line(menu_window, (650, 150), (650, 750), (0, 0, 0), 2)
        
        try:
            cv2.imshow("Game Menu", menu_window)
        except Exception as e:
            print(f"Error showing menu: {e}")
        return "menu"

    def run(self):
        print("Starting camera capture...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return
            
        print("Camera opened successfully")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Starting game loop...")
        
        # Show menu first
        self.show_game_menu()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            frame_display = frame.copy()
            elapsed_time = 0

            if self.game_state == "menu":
                # Keep showing menu until Enter is pressed
                self.show_game_menu()

            elif self.game_state == "playing":
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        nose_pos = self.get_nose_position(face_landmarks, frame.shape[1], frame.shape[0])
                        canvas_x = int((nose_pos[0] / frame.shape[1]) * self.canvas_width)
                        canvas_y = int((nose_pos[1] / frame.shape[0]) * self.canvas_height)
                        smoothed_pos = self.smooth_nose_position((canvas_x, canvas_y))

                        if self.prev_nose_pos:
                            self.draw_smooth_line(self.prev_nose_pos, smoothed_pos)

                        self.prev_nose_pos = smoothed_pos
                else:
                    cv2.putText(frame_display, "Face not detected!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.prev_nose_pos = None
                    self.nose_history.clear()

                elapsed_time = time.time() - self.start_time - self.total_pause_time
                time_left = max(0, int(self.duration - elapsed_time))

                cv2.putText(frame_display, f"Time Left: {time_left}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(frame_display, f"Draw: {self.current_object}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame_display, "P-Pause C-Clear S-Snapshot ENTER-Submit", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                if elapsed_time >= self.duration:
                    self.similarity_score = self.calculate_similarity()
                    self.game_state = "done"

            elif self.game_state == "paused":
                cv2.putText(frame_display, "GAME PAUSED", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(frame_display, "Press P to resume", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            elif self.game_state == "done":
                cv2.putText(frame_display, f"Game Over! Score: {self.similarity_score}%", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 128, 255), 3)
                cv2.putText(frame_display, f"Press R to restart, Q to return to menu", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                
                # Add high score if qualified (only once)
                if not hasattr(self, 'score_added'):
                    self.score_added = False
                
                # Check if this score qualifies for high score
                min_score = min([s["score"] for s in self.high_scores], default=0)
                qualifies = len(self.high_scores) < 10 or self.similarity_score > min_score
                
                if not self.score_added and qualifies:
                    self.add_high_score(self.current_player_name, self.similarity_score, self.current_object)
                    print(f"High score added! {self.current_player_name}: {self.similarity_score}%")
                    self.score_added = True
                elif not self.score_added:
                    print(f"Score: {self.similarity_score}% (not a high score, minimum: {min_score}%)")
                    self.score_added = True
                self.name_entered = False # Reset name entered flag for new game

            # Display
            try:
                if self.game_state == "menu":
                    # Only show menu window when in menu state
                    pass  # Menu is handled by show_game_menu()
                else:
                    # Create full screen side-by-side display that fits the screen
                    # Get actual screen dimensions
                    screen_width = 1600  # Reduced from 1920 to fit more screens
                    screen_height = 900  # Reduced from 1080 to fit more screens
                    
                    # Calculate dimensions for side-by-side layout
                    half_width = screen_width // 2
                    
                    # Resize frame to half screen width
                    frame_resized = cv2.resize(frame_display, (half_width, screen_height))
                    
                    # Resize canvas to match
                    canvas_resized = cv2.resize(self.canvas, (half_width, screen_height))
                    
                    # Add reference image to top right of canvas
                    ref_img = self.generate_reference_image(self.current_object)
                    small_ref = cv2.resize(ref_img, (150, 100))  # Larger reference image
                    # Place in top right corner of canvas with better positioning
                    canvas_resized[20:120, half_width-170:half_width-20] = small_ref
                    # Add label with better visibility
                    cv2.putText(canvas_resized, "REFERENCE", (half_width-170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Create combined display
                    combined_display = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
                    
                    # Place camera feed on the left
                    combined_display[:, :half_width] = frame_resized
                    
                    # Place canvas on the right
                    combined_display[:, half_width:] = canvas_resized
                    
                    # Add separator line
                    cv2.line(combined_display, (half_width, 0), (half_width, screen_height), (0, 0, 0), 3)
                    
                    # Add labels
                    cv2.putText(combined_display, "Camera Feed", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(combined_display, "Drawing Canvas", (half_width + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Create window with proper flags for full screen
                    cv2.namedWindow("Nose Draw Game", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Nose Draw Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("Nose Draw Game", combined_display)
                    
            except Exception as e:
                print(f"Error displaying windows: {e}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('\r') or key == 13:  # Enter key
                if self.game_state == "menu":
                    print("Starting game...")
                    # Show name input dialog before starting game
                    if not self.name_entered:
                        player_name = self.show_name_input_dialog()
                        self.current_player_name = player_name
                        self.name_entered = True
                        print(f"Player: {player_name}")
                    
                    self.start_time = time.time()
                    self.game_state = "playing"
                    self.canvas[:] = 255
                    self.prev_nose_pos = None
                    self.nose_history.clear()
                    self.total_pause_time = 0
                    self.snapshots.clear()
                    self.snapshot_times.clear()
                    self.score_added = False  # Reset score added flag
                    # Close menu window when starting game
                    cv2.destroyWindow("Game Menu")
                elif self.game_state == "playing":
                    print("Submitting drawing...")
                    self.similarity_score = self.calculate_similarity()
                    self.game_state = "done"
            elif key == ord('p'):
                if self.game_state == "playing":
                    self.pause_game()
                elif self.game_state == "paused":
                    self.resume_game()
            elif key == ord('c') and self.game_state == "playing":
                self.clear_canvas()
            elif key == ord('s') and self.game_state == "playing":
                self.take_snapshot()
            elif key == ord('v'):
                if self.game_state == "menu":
                    self.game_state = self.show_snapshot_menu()
            elif key == ord('r'):
                self.canvas[:] = 255
                self.current_object = random.choice(self.objects)
                self.reference_image = self.generate_reference_image(self.current_object)
                self.prev_nose_pos = None
                self.nose_history.clear()
                self.similarity_score = 0
                self.start_time = None
                self.total_pause_time = 0
                self.pause_start_time = None
                self.snapshots.clear()
                self.snapshot_times.clear()
                self.current_player_name = ""  # Reset for new game
                self.name_entered = False  # Reset name entered flag
                self.score_added = False  # Reset score added flag
                self.game_state = "menu"
            elif key == ord('q'):
                # Return to menu instead of quitting
                self.game_state = "menu"
                self.current_player_name = ""  # Reset for new game
                self.name_entered = False  # Reset name entered flag
                self.score_added = False  # Reset score added flag
            elif key == 27:  # ESC key
                print("Quitting game...")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Game ended")

if __name__ == "__main__":
    try:
        print("Starting Nose Draw Game...")
        NoseDrawingGame().run()
    except Exception as e:
        print(f"Error running game: {e}")
        import traceback
        traceback.print_exc()
