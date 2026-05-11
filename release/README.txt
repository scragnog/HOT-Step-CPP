==============================================================
  HOT-Step 9000 — High-Performance AI Music Generation
==============================================================

QUICK START
-----------
1. Extract this folder anywhere you like.
2. Double-click "HOT-Step.bat" to start.
3. Your browser will open to http://localhost:3001 automatically.
4. On first launch, go to Settings > Model Manager to download
   the AI models (~7 GB). You'll need an internet connection.

That's it! No installation required.


REQUIREMENTS
------------
- Windows 10/11 (64-bit)
- NVIDIA GPU with recent drivers (RTX 2060 or newer recommended)
  * CPU-only mode works but is significantly slower
- ~10 GB free disk space (for models + generated audio)
- Internet connection for first-run model downloads


GPU SUPPORT
-----------
This release includes CUDA support for NVIDIA GPUs:
  - RTX 2000 series (Turing)
  - RTX 3000 series (Ampere)
  - RTX 4000 series (Ada Lovelace)
  - RTX 5000 series (Blackwell)

Make sure your NVIDIA drivers are up to date:
  https://www.nvidia.com/en-us/drivers/


STEM SEPARATION (Optional)
---------------------------
Stem separation requires additional runtime files (~1.3 GB).
When you first use Stem Studio, you'll be prompted to download
them from the Model Manager.

These files include ONNX Runtime and cuDNN libraries required
for the neural network stem separator.


CONFIGURATION
-------------
To customize settings, copy ".env.example" to ".env" and edit it.
Most settings can also be changed from the Settings page in the app.


FOLDER STRUCTURE
----------------
  HOT-Step.bat        — Launch the application
  runtime/            — Node.js runtime (do not modify)
  engine/             — C++ inference engine
  server/             — Application server
  ui/                 — Web interface
  models/             — AI model files (downloaded on first run)
  adapters/           — LoRA adapters (optional)
  Essentia/           — Audio analysis tool
  noise_samples/      — Noise profiles for denoising (optional)


TROUBLESHOOTING
---------------
Q: The app won't start.
A: Make sure you extracted the FULL zip. Don't move individual
   files. Try running HOT-Step.bat from an elevated command prompt.

Q: Generation is very slow.
A: Check that your NVIDIA drivers are installed. The app will
   fall back to CPU if no GPU is detected. See Settings > Health
   for GPU status.

Q: No sound / playback issues.
A: Make sure your browser allows audio playback. Try a different
   browser (Chrome or Edge recommended).

Q: Model download fails.
A: Check your internet connection. Downloads can be resumed —
   just click the download button again.


LICENSE & CREDITS
-----------------
Built on ACE-Step (MIT License) — https://github.com/ace-step
GGML inference framework — https://github.com/ggml-org/ggml

For support and updates:
  https://github.com/scragnog/HOT-Step-CPP
