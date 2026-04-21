# Models

Place your GGUF model files here. You need one of each type:

| Type | Example | Role |
|------|---------|------|
| LM | acestep-5Hz-lm-4B-Q8_0.gguf | Generates lyrics and audio codes |
| Text encoder | Qwen3-Embedding-0.6B-Q8_0.gguf | Encodes captions for the DiT |
| DiT | acestep-v15-turbo-Q8_0.gguf | Renders audio codes into sound |
| VAE | vae-BF16.gguf | Decodes latents to 48kHz stereo audio |

Download from: https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/tree/main

Or use the download script:

```bash
./models.sh
```
