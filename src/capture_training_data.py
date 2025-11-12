import cv2
import os
from datetime import datetime
import time

def create_gesture_folder(base_path, gesture_name):
    """Create a folder for the gesture if it doesn't exist."""
    folder_path = os.path.join(base_path, gesture_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def main():
    # Configuration
    BASE_OUTPUT_DIR = "../public/training-data"  # Change this to your desired path
    
    # Create base directory if it doesn't exist
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Variables
    gesture_name = None
    capture_mode = False
    image_count = 0
    capture_interval = 0.1  # 100ms between captures (10 fps)
    last_capture_time = 0
    
    print("="*60)
    print("GESTURE TRAINING DATA CAPTURE TOOL")
    print("="*60)
    print("\nControls:")
    print("  'g' - Enter gesture name (required before capturing)")
    print("  'c' - Start/Stop continuous capture mode")
    print("  SPACE - Capture single image")
    print("  'r' - Reset and enter new gesture name")
    print("  '+' - Increase capture speed")
    print("  '-' - Decrease capture speed")
    print("  'q' - Quit")
    print("="*60)
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Warning: Failed to read frame")
            continue
        
        # Create display frame with info
        display_frame = frame.copy()
        
        # Display current settings
        info_y = 30
        cv2.putText(display_frame, f"Gesture: {gesture_name if gesture_name else 'NOT SET'}", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if gesture_name else (0, 0, 255), 2)
        info_y += 30
        
        cv2.putText(display_frame, f"Mode: {'CAPTURING' if capture_mode else 'PAUSED'}", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if capture_mode else (255, 255, 255), 2)
        info_y += 30
        
        cv2.putText(display_frame, f"Images: {image_count}", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        
        cv2.putText(display_frame, f"Speed: {1/capture_interval:.1f} fps", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Auto-capture in continuous mode
        current_time = time.time()
        if capture_mode and gesture_name and (current_time - last_capture_time) >= capture_interval:
            folder_path = create_gesture_folder(BASE_OUTPUT_DIR, gesture_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{gesture_name}_{timestamp}.jpg"
            filepath = os.path.join(folder_path, filename)
            
            cv2.imwrite(filepath, frame)
            image_count += 1
            last_capture_time = current_time
            
            # Visual feedback
            cv2.putText(display_frame, "CAPTURED!", (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imshow('Gesture Capture', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('g') or key == ord('r'):
            # Enter new gesture name
            capture_mode = False
            cv2.destroyWindow('Gesture Capture')
            
            print("\n" + "="*60)
            gesture_input = input("Enter gesture name (e.g., 'thumbs_up', 'peace', 'fist'): ").strip()
            
            if gesture_input:
                gesture_name = gesture_input
                image_count = 0
                print(f"Gesture set to: {gesture_name}")
                print(f"Images will be saved to: {os.path.join(BASE_OUTPUT_DIR, gesture_name)}")
            else:
                print("No gesture name entered. Please try again.")
            print("="*60 + "\n")
        
        elif key == ord('c'):
            # Toggle continuous capture mode
            if gesture_name:
                capture_mode = not capture_mode
                status = "STARTED" if capture_mode else "STOPPED"
                print(f"\nContinuous capture {status}")
                if capture_mode:
                    last_capture_time = time.time()
            else:
                print("\nPlease set a gesture name first (press 'g')")
        
        elif key == ord(' '):
            # Single capture
            if gesture_name:
                folder_path = create_gesture_folder(BASE_OUTPUT_DIR, gesture_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{gesture_name}_{timestamp}.jpg"
                filepath = os.path.join(folder_path, filename)
                
                cv2.imwrite(filepath, frame)
                image_count += 1
                print(f"Captured: {filename}")
            else:
                print("\nPlease set a gesture name first (press 'g')")
        
        elif key == ord('+') or key == ord('='):
            # Increase speed (decrease interval)
            capture_interval = max(0.05, capture_interval - 0.05)
            print(f"\nCapture speed: {1/capture_interval:.1f} fps")
        
        elif key == ord('-') or key == ord('_'):
            # Decrease speed (increase interval)
            capture_interval = min(2.0, capture_interval + 0.05)
            print(f"\nCapture speed: {1/capture_interval:.1f} fps")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"Capture session completed!")
    print(f"Total images captured: {image_count}")
    if gesture_name:
        print(f"Saved to: {os.path.join(BASE_OUTPUT_DIR, gesture_name)}")
    print("="*60)

if __name__ == "__main__":
    main()
