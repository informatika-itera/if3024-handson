# Mengimport library yang diperlukan untuk mendeteksi respirasi
import os
import requests
from tqdm import tqdm
import platform 
import subprocess
import mediapipe as mp
import re
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from datetime import timedelta
from mediapipe.framework.formats import landmark_pb2
import matplotlib
matplotlib.use('Agg')

def download_model():
    """
    Fungsi ini berguna untuk mengunduh model deteksi wajah dari MediaPipe. 
    """
    # Membuat direktori untuk menyimpan model jika belum ada
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")

    # Mmeriksa apakah file sudah ada dan tidak kosong
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"File model {filename} sudah terunduh, melewati proses pengunduhan.")
        return filename
    # Mengunduh file model dengan progress bar dari library tqdm
    try: 
        print(f"Mengunduh model ke {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Memastikan tidak ada error dalam pengunduhan

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # Ukuran blok untuk pengunduhan

        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)

        # Memeriksa file yang terunduh
        if os.path.getsize(filename) == 0:
            raise ValueError(f"File {filename} terunduh kosong.")

        print("Model berhasil diunduh.")
        return filename
    except Exception as e:
        print(f"Terjadi kesalahan saat mengunduh model: {e}")
        if os.path.exists(filename):
            os.remove(filename) # Menghapus file yang tidak valid
        raise 

def check_gpu():
    """
    Fungsi ini berguna untuk memeriksa apakah GPU tersedia dan mendukung CUDA.
    """
    system = platform.system()
    print(f"System: {system}")
    # Memeriksa untuk ketersediaan NVIDIA GPU
    if system == "Linux" or system == "Windows":
        try:
            nvidia_output = subprocess.check_outpu(['nvidia-smi']).decode('utf-8')
            return "NVIDIA"
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("NVIDIA GPU tidak ditemukan.")
            return "CPU"
    
    # Memeriksa untuk ketersedian Apple MLX  
    elif system == "Darwin":
        try: 
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU: {cpu_info}")
            if "Apple" in cpu_info:
                return "MLX"
        except subprocess.CalledProcessError:
            pass 
    return "CPU"

def enhance_roi(roi):
    """
    Fungsi ini digunakan untuk meningkatkan ROI (Region of Interest) dengan teknik image processing.
    Args:
        roi: Region of Interest dari frame video
    Returns: 
        numpy.ndarray:     
    """
    if roi is None or roi.size == 0: 
        raise ValueError("Empty ROI provided.")
    # Mengubah ROI ke grayscale
    try: 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Menggunakan CLAHE untuk meningkatkan kontras
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Menggunakan edge enhancement
        return enhanced 
    except cv2.error as e: 
        raise ValueError(f"Error in enhancing ROI: {str(e)}")

def get_initial_roi(image, landmarker, x_size=100, y_size=150, shift_x=0, shift_y=0):
    """
    Fungsi ini berguna untuk mendapatkan ROI awal berdasarkan posisi bahu menggunakan pose detection.
    Args: 
        image: Frame video input
        landmarker: Model pose detector
        x_size: Jarak piksel dari titik tengah ke tepi kiri/kanan
        y_size: Jarak piksel dari titik tengah ke tepi atas/bawah
        shift_x: Pergeseran horizontal kotak (negatif=kiri, positif=kanan)
        shift_y: Pergeseran vertikal kotak (negatif=atas, positif=bawah)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Membuat mediapipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    # Mendeteksi landmarks 
    detection_result = landmarker.detect(mp_image)

    if not detection_result.pose_landmarks:
        raise ValueError("Tidak ada pose landmarks yang terdekteksi di frame awal.")
    
    landmarks = detection_result.pose_landmarks[0]

    # Mengambi koordinat dari bahu kiri dan kanan subjek
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    # Menghitung titik tengah antara bahu kiri dan kanan
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)

    # Mengapplikasikan pergeseran/shift terhadap titik tengah
    center_x += shift_x
    center_y += shift_y

    # Menghitung batasan kotak ROI dari titik tengah dan ukuran yang ditentukan
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)

    # Memvalidasi ukuran ROI
    if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
        raise ValueError("Ukuran ROI tidak valid.")
    
    return(left_x, right_x, top_y, bottom_y)

def prepare_plot():
    """
    Mempersiapkan plot matplotlib untuk visualisasi gerakan bahu.
    Returns:
        tuple: (figure, axis) untuk plotting
    """
    # Create smaller figure with transparent background
    fig = plt.figure(figsize=(4, 3), facecolor='none')
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.7)  # Semi-transparent background
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Shoulder Movement')
    ax.grid(True, alpha=1)  # Lighter grid
    return fig, ax

def process_video(landmarker, video_path, max_seconds=20, x_size=300, y_size=250, shift_x=0, shift_y=0):
    """
    Memproses video untuk melacak gerakan bahu.
    Menggunakan optical flow dan pose detection untuk tracking.
    
    Args:
        landmarker: Model pose detector
        video_path: Path ke file video
        max_seconds: Durasi maksimum video yang diproses
        x_size: Lebar ROI
        y_size: Tinggi ROI
        shift_x: Pergeseran horizontal ROI
        shift_y: Pergeseran vertikal ROI
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)
    
    # Initialize video writer with original frame size
    output_path = 'media/toby-shoulder-track.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
 
           # Prepare plot
    fig, ax = prepare_plot()
    timestamps = []
    y_positions = []
    
    # Read first frame and get ROI
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame!")
    
    try:
        # Mendapatkan ROI awal dari frame pertama
        roi_coords = get_initial_roi(first_frame, landmarker, 
                                   x_size=x_size, y_size=y_size,
                                   shift_x=shift_x, shift_y=shift_y)
        left_x, top_y, right_x, bottom_y = roi_coords
        
        # Inisialisasi tracking dengan Optical Flow
        old_frame = first_frame.copy()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize ROI and feature detection
        roi = old_gray[top_y:bottom_y, left_x:right_x]
        features = cv2.goodFeaturesToTrack(roi, 
                                         maxCorners=60,
                                         qualityLevel=0.15,
                                         minDistance=3,
                                         blockSize=7)
        
        if features is None:
            raise ValueError("No features found to track!")
            
        # Menyesuaikan koordinat fitur ke frame penuh
        features = np.float32(features)
        
        # Adjust coordinates to full frame
        features[:,:,0] += left_x
        features[:,:,1] += top_y
        
        # LK params for better tracking
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        frame_count = 0
        pbar = tqdm(total=max_frames, desc='Processing frames')
        
        # Loop utama pemrosesan video
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if len(features) > 10:  # Ensure we have enough features
                # Calculate optical flow
                new_features, status, error = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, features, None, **lk_params)
                
                # Select good points
                good_old = features[status == 1]
                good_new = new_features[status == 1]
                
                # Draw tracks and calculate movement
                mask = np.zeros_like(frame)
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
                
                frame = cv2.add(frame, mask)
                
                # Update tracking points
                if len(good_new) > 0:
                    avg_y = np.mean(good_new[:, 1])
                    y_positions.append(avg_y)
                    timestamps.append(frame_count / fps)
                    features = good_new.reshape(-1, 1, 2)
                    
                    # Update plot
                    ax.clear()
                    ax.set_facecolor('none')
                    ax.patch.set_alpha(0.7)
                    ax.plot(timestamps, y_positions, 'g-', linewidth=2)
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Y Position (pixels)')
                    ax.set_title('Shoulder Movement')
                    ax.grid(True, alpha=0.3)
                    
                    # Set consistent axis limits
                    if len(timestamps) > 1:
                        ax.set_xlim(0, max_seconds)
                        y_min, y_max = min(y_positions), max(y_positions)
                        y_range = y_max - y_min
                        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
                
                    # Convert plot to image
                    fig.canvas.draw()
                    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # Resize plot to desired size (e.g., 1/4 of the frame width)
                    plot_height = int(height * 0.3)  # 30% of frame height
                    plot_width = int(width * 0.3)    # 30% of frame width
                    plot_img = cv2.resize(plot_img, (plot_width, plot_height))
                    
                    # Create mask for plot background
                    plot_gray = cv2.cvtColor(plot_img, cv2.COLOR_RGB2GRAY)
                    _, mask = cv2.threshold(plot_gray, 0, 255, cv2.THRESH_BINARY)
                    mask = mask.astype(bool)
                    
                    # Calculate position for overlay (top right corner with padding)
                    padding = 20
                    y_offset = padding
                    x_offset = width - plot_width - padding
                    
                    # Overlay plot on frame
                    roi = frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width]
                    roi[mask] = plot_img[mask]
                    frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = roi
                    
                    # Write frame
                    out.write(frame)
            else:
                # If we lose too many features, detect new ones
                roi = frame_gray[top_y:bottom_y, left_x:right_x]
                features = cv2.goodFeaturesToTrack(roi, 
                                                 maxCorners=100,
                                                 qualityLevel=0.01,
                                                 minDistance=7,
                                                 blockSize=7)
                if features is not None:
                    features = features + np.array([[left_x, top_y]], dtype=np.float32)
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 2)
            
            # Update for next frame
            old_gray = frame_gray.copy()
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        plt.close(fig)
        cap.release()
        out.release()
        
        return timestamps, y_positions
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        cap.release()
        out.release()
        raise

def plot_shoulder_movement(timestamps, y_positions):
    """
    Membuat plot gerakan bahu terhadap waktu.
    Args:
        timestamps: Array waktu
        y_positions: Array posisi y bahu
    """
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, y_positions, label='Average Y Position', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Chest/Shoulder Movement Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Fungsi utama program.
    Menginisialisasi model dan memproses video untuk tracking gerakan bahu.
    """
    detector_image = None
    try:
        # 1. Download the model
        model_path = download_model()
        
        # 2. Prepare the pose landmarkers
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        gpu_check = check_gpu()
        
        if gpu_check == "NVIDIA":
            delegate = BaseOptions.Delegate.GPU
        else:
            delegate = BaseOptions.Delegate.CPU
        
        # Create landmarker for initial frame (IMAGE mode)
        options_image = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=delegate
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        
        # Create detector
        detector_image = PoseLandmarker.create_from_options(options_image)
        
        video_path = 'media/toby-rgb.mp4'
        print("\nProcessing video...")
        
        # Process video
        timestamps, y_positions = process_video(detector_image, video_path,
                                             max_seconds=20,
                                             x_size=300,
                                             y_size=200,
                                             shift_x=0,
                                             shift_y=100)
        
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        if detector_image:
            detector_image.close()

if __name__ == "__main__":
    main()