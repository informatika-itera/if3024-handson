import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import os 
import scipy.signal as signal 
from glob import glob

VIDEO_PATH = os.path.join('media', 'toby-rgb.mp4')

def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

def main():
    # 1. Inisialisasi MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # 2. Memuat video dan melakukan perulangan
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 2.1 Mempersiapkan beberapa variabel
    r_signal, g_signal, b_signal = [], [], []
    f_count = 0
    f_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2.2 Perulangan Utama
    try:
        while cap.isOpened():
            print(f'Processing Frame {f_count}/{f_total}', end='\r')
            ret, frame = cap.read()
            
            ### 3. Mendeteksi area wajah menggunakan mediapipe
            
            ### 3.1 Mengkonversi frame ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            ### 3.2 Memproses frame menggunakan face_detection
            results = face_detection.process(frame_rgb)
            
            if results.detections: # If there are faces detected
                for detection in results.detections: # Loop through all the detected faces
                    ### 3.3 Mendapatkan bounding box dari wajah
                    bbox = detection.location_data.relative_bounding_box
                    ### 3.4 Mendapatkan lebar dan tinggi frame
                    h, w, _ = frame.shape
                    ### 3.5 Mengkonversi bounding box ke koordinat piksel
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    ### 3.6 Mengkonversi lebar dan tinggi bounding box ke koordinat piksel
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    
                    ### 3.7 Melakukan penyesuaian pada bounding box
                    bbox_size_from_center = 70
                    
                    bbox_center_x = x + width // 2
                    bbox_center_y = y + height // 2
                    new_x = bbox_center_x - bbox_size_from_center
                    new_y = bbox_center_y - bbox_size_from_center
                    new_width = bbox_size_from_center * 2
                    new_height = bbox_size_from_center * 2
                    
                    ### 3.8 Menggambar bounding box pada frame
                    cv2.rectangle(frame, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 255, 0), 2)
                    
                    ### 4 Mendapatkan nilai rata-rata piksel dari ROI dan menambahkannya ke signal
                    roi = frame[new_y:new_y+new_height, new_x:new_x+new_width]
                    r_signal.append(np.mean(roi[:, :, 0]))
                    g_signal.append(np.mean(roi[:, :, 1]))
                    b_signal.append(np.mean(roi[:, :, 2]))
            
            if not ret:
                break
            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            f_count += 1
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
    
    # 5. Menampilkan grafik sinyal
    ax, fig = plt.subplots(3, 1, figsize=(20, 10))
    fig[0].plot(r_signal, color='red')
    fig[0].set_title('Red Signal')
    fig[1].plot(g_signal, color='green')
    fig[1].set_title('Green Signal')
    fig[2].plot(b_signal, color='blue')
    fig[2].set_title('Blue Signal')
    plt.tight_layout()
    plt.show()
    
    
    # 6. Menghitung rPPG menggunakan Metode POS
    rgb_signals = np.array([r_signal, g_signal, b_signal])
    rgb_signals = rgb_signals.reshape(1, 3, -1)
    rppg_signal = cpu_POS(rgb_signals, fps=30)
    rppg_signal = rppg_signal.reshape(-1)
    
    # 6.1 Menampilkan grafik Sinyal rPPG
    plt.figure(figsize=(20, 5))
    plt.plot(rppg_signal, color='black')
    plt.title('rPPG Signal')
    plt.tight_layout()
    plt.show()
    
    # 7. Memfilter Sinyal rPPG
    fs = 30; lowcut = 0.9; highcut = 2.4; order = 3
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    filtered_rppg = signal.filtfilt(b, a, rppg_signal)
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 6))
    ax[0].plot(rppg_signal, color='black')
    ax[0].set_title('rPPG Signal - Before Filtering')
    ax[1].plot(filtered_rppg, color='black')
    ax[1].set_title('rPPG Signal - After Filtering')
    plt.tight_layout()
    plt.show()
    
    
    # 8. Menghitung Heart Rate
    
    ## 8.1 Normalisasi Sinyal
    filtered_rppg = (filtered_rppg - np.mean(filtered_rppg)) / np.std(filtered_rppg)
    
    
    ## 8.2 Mencari puncak sinyal
    peaks, _ = signal.find_peaks(
        x=filtered_rppg,
        prominence=0.5,
    )
    
    ## 8.3 Menghitung heart rate
    heart_rate = 60 * len(peaks) / (len(filtered_rppg) / fs)
    
    ## 8.4 Menampilkan grafik puncak sinyal
    plt.figure(figsize=(20, 5))
    plt.plot(filtered_rppg, color='black')
    plt.plot(peaks, filtered_rppg[peaks], 'x', color='red')
    plt.title(f'Heart Rate: {heart_rate:.2f}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()