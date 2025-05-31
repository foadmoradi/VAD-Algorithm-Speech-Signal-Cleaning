import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import get_window

def segmentation_function(signal, W=256, shift_percentage=0.4, window_frame=None):
    """
    Segment signal into overlapping windowed frames.

    Parameters:
        signal (ndarray): Input signal (1D array).
        W (int): Window length in samples (default=256).
        shift_percentage (float): Shift percentage (default=0.4).
        window_frame (ndarray): Window vector (default=Hamming).

    Returns:
        ndarray: Segmented and windowed signal (2D array, each column is a frame).
    """
    if window_frame is None:
        window_frame = get_window('hamming', W)
    window_frame = window_frame.reshape(-1, 1)  # Column vector
    
    shift_length = int(W * shift_percentage)
    L = len(signal)
    N = int((L - W) / shift_length) + 1  # Number of segments
    
    # Create index matrix
    indices = np.arange(W).reshape(-1, 1) + np.arange(N) * shift_length
    # Ensure indices are within bounds
    indices = np.clip(indices, 0, L - 1).astype(int)
    
    # Apply window to each segment
    segmented = signal[indices] * window_frame
    return segmented

def adding_overlaps(new_signal, y_signal_phase=None, win_length=None, shift_length=None):
    """
    Reconstruct signal from frequency-domain segments.

    Parameters:
        new_signal (ndarray): Magnitude spectrum of segments (each column is a frame).
        y_signal_phase (ndarray): Phase spectrum (same shape as new_signal).
        win_length (int): Window length (default=2*new_signal.shape[0]).
        shift_length (int): Shift length (default=win_length//2).

    Returns:
        ndarray: Reconstructed time-domain signal.
    """
    if y_signal_phase is None:
        y_signal_phase = np.angle(new_signal)
    nfft = new_signal.shape[0]
    
    if win_length is None:
        win_length = nfft * 2
    if shift_length is None:
        shift_length = win_length // 2
    shift_length = int(shift_length)
    
    # Form complex spectrum
    spec = new_signal * np.exp(1j * y_signal_phase)
    
    # Reconstruct full FFT spectrum
    if win_length % 2 == 1:  # Odd
        full_spec = np.vstack((spec, np.flipud(np.conj(spec[1:, :]))))
    else:  # Even
        full_spec = np.vstack((spec, np.flipud(np.conj(spec[1:-1, :]))))
    
    # Overlap-add reconstruction
    frame_number = full_spec.shape[1]
    total_length = (frame_number - 1) * shift_length + win_length
    sig = np.zeros(total_length)
    
    for i in range(frame_number):
        start = i * shift_length
        end = start + win_length
        if end > total_length:
            break
        frame = np.real(np.fft.ifft(full_spec[:, i], win_length))
        sig[start:end] += frame
    return sig

def vad_function(signal, noise, noise_counter=0, noise_margin=3, hangover=8):
    """
    Voice Activity Detection using spectral distance.

    Parameters:
        signal (ndarray): Current frame's magnitude spectrum.
        noise (ndarray): Estimated noise magnitude spectrum.
        noise_counter (int): Count of previous noise frames.
        noise_margin (float): Threshold in dB (default=3).
        hangover (int): Hangover period (default=8).

    Returns:
        tuple: (flag_noise, flag_speech, noise_counter, distance)
    """
    spectral_dist = 20 * (np.log10(signal + 1e-10) - np.log10(noise + 1e-10))
    spectral_dist[spectral_dist < 0] = 0
    distance = np.mean(spectral_dist)
    
    if distance < noise_margin:
        flag_noise = 1
        noise_counter += 1
    else:
        flag_noise = 0
        noise_counter = 0
        
    flag_speech = 0 if noise_counter > hangover else 1
    return flag_noise, flag_speech, noise_counter, distance

def speech_enhancement(signal, fs, noise_segment=0.25):
    """
    Speech enhancement using spectral subtraction.

    Parameters:
        signal (ndarray): Input noisy speech signal.
        fs (int): Sampling frequency.
        noise_segment (float): Length of noise segment in seconds (default=0.25).

    Returns:
        ndarray: Enhanced speech signal.
    """
    W = int(0.025 * fs)  # Window length (25 ms)
    shift_percentage = 0.4
    shift_length = int(W * shift_percentage)
    nfft = W
    gamma = 1  # Magnitude spectral subtraction
    window = get_window('hamming', W)
    
    # Segment signal
    y = segmentation_function(signal, W, shift_percentage, window)
    Y = np.fft.fft(y, nfft, axis=0)
    
    # Extract magnitude and phase
    half_nfft = nfft // 2 + 1 if nfft % 2 == 0 else (nfft + 1) // 2
    Y_mag = np.abs(Y[:half_nfft, :]) ** gamma
    Y_phase = np.angle(Y[:half_nfft, :])
    
    # Initial noise estimate
    num_initial_noise_frames = int((noise_segment * fs - W) / shift_length) + 1
    N = np.mean(Y_mag[:, :num_initial_noise_frames], axis=1).reshape(-1, 1)
    
    # Parameters
    noise_residual_max = np.zeros((half_nfft, 1))
    noise_counter = 0
    noise_length = 9  # Smoothing factor
    beta = 0.03
    num_frames = Y_mag.shape[1]
    X = np.zeros_like(Y_mag)
    
    # Smooth magnitude
    Y_smooth = Y_mag.copy()
    for i in range(1, num_frames - 1):
        Y_smooth[:, i] = np.mean(Y_mag[:, i-1:i+2], axis=1)
    
    # Process each frame
    for i in range(num_frames):
        mag_frame = Y_mag[:, i] ** (1/gamma)
        noise_frame = N[:, 0] ** (1/gamma)
        flag_noise, flag_speech, noise_counter, _ = vad_function(
            mag_frame, noise_frame, noise_counter
        )
        
        if flag_speech == 0:  # Noise frame
            N = (noise_length * N + Y_mag[:, i:i+1]) / (noise_length + 1)
            residue = Y_smooth[:, i:i+1] - N
            noise_residual_max = np.maximum(noise_residual_max, residue)
            X[:, i] = beta * Y_mag[:, i]
        else:  # Speech frame
            D = Y_smooth[:, i] - N[:, 0]
            # Residual noise reduction
            if i > 0 and i < num_frames - 1:
                for j in range(len(D)):
                    if D[j] < noise_residual_max[j, 0]:
                        neighbors = [
                            D[j],
                            Y_smooth[j, i-1] - N[j, 0],
                            Y_smooth[j, i+1] - N[j, 0]
                        ]
                        D[j] = min(neighbors)
            X[:, i] = np.maximum(D, 0)
    
    # Reconstruct signal
    output = adding_overlaps(
        X ** (1/gamma), Y_phase, W, shift_length
    )
    return output

if __name__ == "__main__":
    # Read audio file
    file_path = 'M02_LT_z01.wav'
    x, fs = sf.read(file_path)
    
    # Enhance speech
    output = speech_enhancement(x, fs)
    
    # Plot results
    t_orig = np.arange(len(x)) / fs * 1000
    t_output = np.arange(len(output)) / fs * 1000
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_orig, x)
    plt.grid(True)
    plt.title('Original Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.xlim(0, t_orig[-1])
    
    plt.subplot(2, 1, 2)
    plt.plot(t_output, output)
    plt.grid(True)
    plt.title('Cleaned Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.xlim(0, t_output[-1])
    
    plt.tight_layout()
    plt.show()
    
    # Play audio
    print("Playing original audio...")
    sd.play(x, fs)
    sd.wait()
    
    print("Playing enhanced audio...")
    sd.play(output, fs)
    sd.wait()