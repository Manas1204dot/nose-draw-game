import cv2
import numpy as np
import random

def generate_reference_image(obj_type, canvas_width=800, canvas_height=600):
    ref = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    center = (canvas_width // 2, canvas_height // 2)
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

def show_all_shapes():
    shapes = ["window", "star", "triangle", "cloud", "tree", "house", "flower"]
    
    # Create a grid to display all shapes
    rows, cols = 2, 4
    cell_width, cell_height = 400, 300
    
    # Create the main display
    display = np.ones((rows * cell_height, cols * cell_width, 3), dtype=np.uint8) * 255
    
    for i, shape in enumerate(shapes):
        row = i // cols
        col = i % cols
        
        # Generate the shape
        shape_img = generate_reference_image(shape, cell_width, cell_height)
        
        # Add shape name
        cv2.putText(shape_img, shape.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Place in the grid
        y_start = row * cell_height
        y_end = (row + 1) * cell_height
        x_start = col * cell_width
        x_end = (col + 1) * cell_width
        
        display[y_start:y_end, x_start:x_end] = shape_img
    
    # Add title
    cv2.putText(display, "All Available Shapes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    cv2.imshow("All Shapes", display)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_all_shapes() 