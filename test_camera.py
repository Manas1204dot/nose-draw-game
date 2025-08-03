import cv2
import numpy as np

def test_camera():
    print("Starting camera test...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully")
    
    # Create a simple test window
    test_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_window, "Test Window", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Test Window", test_window)
    
    print("Test window created. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        cv2.imshow("Camera Feed", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    test_camera() 