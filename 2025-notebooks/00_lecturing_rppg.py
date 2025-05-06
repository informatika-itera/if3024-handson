import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import os
import cv2
import mediapipe as mp

def detect_faces(frame, face_detection):
    # Process the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Default values if no face is detected
    x, y, width, height = 0, 0, 0, 0

    # Extract face bounding box if a face is detected
    if results.detections:
        detection = results.detections[0]  # Get the first detected face
        bboxC = detection.location_data.relative_bounding_box
        
        ih, iw, _ = frame.shape
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        width = int(bboxC.width * iw)
        height = int(bboxC.height * ih)
    
    return x,y,width,height

def main():
    # Load video from `rppg_data/rppg_video_20250506_073842.mov`
    video_path = os.path.join('rppg_data', 'rppg_video_20250506_073842.mov')
    
    # Load Ground Truth CSV from `rppg_data/pulse_markers_20250506_073842.csv`
    gt_path = os.path.join('rppg_data', 'pulse_markers_20250506_073842.csv')
    gt_df = pd.read_csv(gt_path)
    
    # Initialize MediaPipe face detection once
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Iterate through the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f_count = 0
    f_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    r_signal, g_signal, b_signal = [], [], []
    
    try:
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face using the already initialized detector
            x, y, width, height = detect_faces(frame, face_detection)
            
            # Print face bounding box coordinates
            print(f"Frame {f_count}/{f_total}: Face detected at x={x}, y={y}, width={width}, height={height}")
            
            # make the bounding box top part higher
            y = int(y - height * 0.25)
            height = int(height * 1.2)
            
            # Draw rectangle around the face if a face was detected
            if width > 0 and height > 0:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Only process ROI if we have valid dimensions
                if y >= 0 and x >= 0 and y+height <= frame.shape[0] and x+width <= frame.shape[1]:
                    # get the average pixel value of the ROI
                    roi = frame[y:y+height, x:x+width]
                    r_mean = np.mean(roi[:, :, 0])
                    g_mean = np.mean(roi[:, :, 1])
                    b_mean = np.mean(roi[:, :, 2])
                else:
                    # Use previous values or defaults if ROI is invalid
                    r_mean = r_signal[-1] if r_signal else 0
                    g_mean = g_signal[-1] if g_signal else 0
                    b_mean = b_signal[-1] if b_signal else 0
            else:
                # Use previous values or defaults if no face detected
                r_mean = r_signal[-1] if r_signal else 0
                g_mean = g_signal[-1] if g_signal else 0
                b_mean = b_signal[-1] if b_signal else 0
                
            r_signal.append(r_mean)
            g_signal.append(g_mean)
            b_signal.append(b_mean)
                
            # Display the frame with bounding box
            cv2.imshow('Face Detection', frame)
            
            # Display progress
            f_count += 1
            if f_count % 30 == 0:  # Update progress every 30 frames
                print(f"Processing: {f_count}/{f_total} frames ({f_count/f_total*100:.1f}%)")
            
            # Optional: Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error during video processing: {e}")
    
    finally:
        # Ensure resources are released properly
        cap.release()
        cv2.destroyAllWindows()
        # Close MediaPipe resources
        face_detection.close()
    
    # Convert signals to numpy arrays
    r_signal = np.array(r_signal)
    g_signal = np.array(g_signal)
    b_signal = np.array(b_signal)
    
    # Create time array
    time = np.arange(len(r_signal)) / fps
    
    # Plot RGB signals
    plt.figure(figsize=(12, 6))
    plt.plot(time, r_signal, 'r-', label='Red')
    plt.plot(time, g_signal, 'g-', label='Green')
    plt.plot(time, b_signal, 'b-', label='Blue')
    plt.title('RGB Signal from Face ROI')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Pixel Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Save the RGB signals to CSV
    csv_path = os.path.join('rppg_data', 'rppg_video_20250506_073842_rgb.csv')
    df = pd.DataFrame({
        'time': time,
        'red': r_signal,
        'green': g_signal,
        'blue': b_signal
    })
    df.to_csv(csv_path, index=False)
    print(f"RGB signals saved to: {csv_path}")
    
    
if __name__ == "__main__":
    main()