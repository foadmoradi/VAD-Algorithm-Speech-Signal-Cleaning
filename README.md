# VAD-Algorithm-Speech-Signal-Cleaning
VAD is accurate and efficient algorithm for cleaning background noise in speech signals. Here, you find a detailed implementation of the algorithm.

🚀 Just launched a powerful speech enhancement tool from my freelancing work in speech signal processing! This Python implementation cleans noisy recordings using spectral subtraction + voice activity detection (VAD).

🔧 Installation Note:

To load sound and play denoised speech signal you use soundfile and sounddevice paxckages. Don't forget to install them:

pip install soundfile sounddevice

If you get "OSError: PortAudio library not found":

sudo apt install portaudio19-dev  # Ubuntu/Debian

🔍 How it works:
1️⃣ SegmentationFunction:

    Chops audio into 25ms Hamming windows (40% overlap)
    2️⃣ VAD_Function:

    Flags speech/noise using spectral distance + hangover logic
    3️⃣ SpeechEnhancement:

    Estimates noise → subtracts it → suppresses musical artifacts
    4️⃣ AddingOverlaps:

    Reconstructs clean audio via overlap-add

✅ Pros:

    Lightning-fast (O(n) complexity)

    Zero training needed

    Crushes stationary noise (fans/hums)

    Preserves speech clarity

⚠️ Limitations:

    Musical noise in low-SNR

    Struggles with sudden noises

💡 Applications:

    ASR preprocessing

    Podcast/old recording restoration

    Hearing assistive devices

📈 Why it matters: This algorithm demonstrates how classical DSP achieves real-time noise reduction with minimal compute – perfect for edge devices!
