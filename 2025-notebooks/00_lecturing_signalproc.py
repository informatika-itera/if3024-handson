import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import os

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

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the signal
    
    Parameters:
    -----------
    data : array-like
        Input signal to filter
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int, optional
        Filter order, default is 5
    
    Returns:
    --------
    filtered_data : array-like
        Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def main():
    # Load the mean rgb value for POS rppg algorithm
    # path: `rppg_data/rppg_video_20250506_073842_rgb.csv`
    rgb_path = os.path.join('rppg_data', 'rppg_video_20250506_073842_rgb.csv')
    rgb_df = pd.read_csv(rgb_path)
    rgb_signal = rgb_df[['red', 'green', 'blue']].values.T
    fps = 30  # frames per second
    
    # Store RGB means for plotting
    mean_r = rgb_signal[0, :]
    mean_g = rgb_signal[1, :]
    mean_b = rgb_signal[2, :]
    
    # Reshape for POS algorithm
    rgb_signal = rgb_signal.reshape(1, 3, -1)  # Reshape to (1, 3, f)
    
    # Load the ground truth signal (binary format)
    # path: `rppg_data/pulse_markers_20250506_073842.csv`
    # Format: `frame_number,timestamp,pulse_marker (binary)`
    gt_path = os.path.join('rppg_data', 'pulse_markers_20250506_073842.csv')
    gt_df = pd.read_csv(gt_path)
    pulse_marker = gt_df['pulse_marker'].values
    
    # Run the POS algorithm
    pos_signal = cpu_POS(rgb_signal, fps=fps)
    print(f"Berapakah shape pos_signal? {pos_signal.shape}")
    
    # Reshape pos_signal for easier handling (flatten if it's a 2D array)
    pos_signal = pos_signal.flatten()
    
    # Apply bandpass filter to POS signal - default between 1Hz and 2Hz
    # (typical heart rate range is 60-120 BPM = 1-2 Hz)
    lowcut = 0.8  # Hz
    highcut = 2.5  # Hz
    filtered_pos = bandpass_filter(pos_signal, lowcut, highcut, fps)
    
    # Find peaks in the filtered signal
    # Adjust height and distance parameters based on your signal characteristics
    peaks, _ = signal.find_peaks(filtered_pos, height=0.01, distance=fps/2)  # At least 0.5s between peaks
    
    # Count the detected peaks
    peak_count = len(peaks)
    estimated_hr = peak_count * (60 / (len(filtered_pos) / fps))  # Peaks per minute
    
    print(f"Number of peaks detected: {peak_count}")
    print(f"Estimated heart rate: {estimated_hr:.1f} BPM")
    
    # Convert peak indices to time values
    peak_times = peaks / fps
    peak_values = filtered_pos[peaks]
    
    # Create time array
    time = np.arange(len(pos_signal)) / fps  # Convert frames to seconds
    
    # Add markers at pulse locations
    pulse_frames = np.where(pulse_marker == 1)[0]
    
    # Only include pulse frames that are within the length of pos_signal
    valid_pulse_frames = [frame for frame in pulse_frames if frame < len(pos_signal)]
    valid_pulse_times = np.array(valid_pulse_frames) / fps
    valid_pulse_values_pos = np.array([pos_signal[frame] for frame in valid_pulse_frames])
    valid_pulse_values_filtered = np.array([filtered_pos[frame] for frame in valid_pulse_frames])
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Mean RGB signals
    axs[0].plot(time, mean_r, 'r-', label='Red', alpha=0.8)
    axs[0].plot(time, mean_g, 'g-', label='Green', alpha=0.8)
    axs[0].plot(time, mean_b, 'b-', label='Blue', alpha=0.8)
    
    # Add pulse markers to RGB plot
    if len(valid_pulse_times) > 0:
        # Use green channel for marking pulses in RGB plot
        pulse_rgb_values = np.array([mean_g[frame] for frame in valid_pulse_frames])
        axs[0].plot(valid_pulse_times, pulse_rgb_values, 'rx', markersize=8)
    
    axs[0].set_title('RGB Channel Mean Values')
    axs[0].set_ylabel('Pixel Value')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: POS signal
    axs[1].plot(time, pos_signal, 'b-', label='POS Signal')
    
    # Add pulse markers to POS plot
    if len(valid_pulse_times) > 0:
        axs[1].plot(valid_pulse_times, valid_pulse_values_pos, 'rx', markersize=8, label='Pulse Marker')
    
    axs[1].set_title('POS rPPG Signal')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Filtered POS signal
    axs[2].plot(time, filtered_pos, 'g-', label=f'Filtered POS ({lowcut:.1f}-{highcut:.1f} Hz)')
    
    # Add detected peaks to the filtered signal plot
    axs[2].plot(peak_times, peak_values, 'bo', markersize=6, label=f'Detected Peaks ({peak_count})')
    
    # Add pulse markers to filtered POS plot
    if len(valid_pulse_times) > 0:
        axs[2].plot(valid_pulse_times, valid_pulse_values_filtered, 'rx', markersize=8, label='Pulse Marker')
    
    # Update title to include peak count
    axs[2].set_title(f'Bandpass Filtered POS Signal - {peak_count} peaks detected ({estimated_hr:.1f} BPM)')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)
    
    # Add common title and adjust layout
    plt.suptitle('Comparison of RGB, POS and Filtered POS Signals with Ground Truth Pulse Markers', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    
if __name__ == "__main__":
    main()