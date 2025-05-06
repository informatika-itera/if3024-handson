import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime

def countdown(cap, seconds=3):
    """Display a countdown on the screen before starting recording"""
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Error during countdown")
            return False
            
        # Display large countdown number
        h, w = frame.shape[:2]
        text = str(i)
        font_scale = 5
        thickness = 5
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Calculate position to center text
        x = (w - text_width) // 2
        y = (h + text_height) // 2
        
        # Draw text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 255), thickness)
        
        # Add instruction
        instruction = "Get ready..."
        cv2.putText(frame, instruction, (w//2 - 100, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('rPPG Capture Countdown', frame)
        cv2.waitKey(1000)  # Wait for 1 second
    
    return True

def main():
    # Create output directory if it doesn't exist
    output_dir = "rppg_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(output_dir, f"rppg_video_{timestamp}.mov")
    csv_filename = os.path.join(output_dir, f"pulse_markers_{timestamp}.csv")
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    
    # Define frame limit
    max_frames = 600
    
    print("rPPG Video Capture Initializing...")
    print(f"Will capture {max_frames} frames (approximately {max_frames/fps:.1f} seconds)")
    print("Instructions:")
    print("1. Position your left hand in front of the camera for pulse detection")
    print("2. Press SPACE when you feel a pulse beat")
    print("3. Press 'q' to quit the recording early")
    print("\nStarting countdown...")
    
    # Display countdown before starting
    if not countdown(cap, 3):
        cap.release()
        cv2.destroyAllWindows()
        return
        
    # Create VideoWriter object with MJPG codec for .mov files
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
    
    # Open CSV file for writing markers
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'timestamp', 'pulse_marker'])
        
        # Variables for tracking
        frame_count = 0
        pulse_marker = 0
        start_time = time.time()
        
        print("Recording started!")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Get current timestamp
            current_time = time.time() - start_time
            
            # Calculate remaining frames and time
            frames_remaining = max_frames - frame_count
            time_remaining = frames_remaining / fps
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Reset pulse marker for new frame
            pulse_marker = 0
            
            # Check for spacebar press to mark pulse
            if key == 32:  # ASCII code for spacebar
                pulse_marker = 1
                
            # Check for quit command
            if key == ord('q'):
                break
            
            # Add visual indicator when pulse is marked
            if pulse_marker == 1:
                cv2.putText(frame, "PULSE DETECTED!", (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Add frame counter, countdown, and instructions
            cv2.putText(frame, f"Frame: {frame_count}/{max_frames}", (30, frame_height - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Remaining: {time_remaining:.1f}s", (30, frame_height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, "Press SPACE to mark pulse beat", (30, frame_height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Write the frame to video
            out.write(frame)
            
            # Write data to CSV
            csv_writer.writerow([frame_count, f"{current_time:.3f}", pulse_marker])
            
            # Display the frame
            cv2.imshow('rPPG Capture (Press q to quit)', frame)
            
            # Increment frame counter
            frame_count += 1
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Recording completed! {frame_count} frames captured.")
    print(f"Video saved as {video_filename}")
    print(f"Pulse markers saved as {csv_filename}")

if __name__ == "__main__":
    main()
