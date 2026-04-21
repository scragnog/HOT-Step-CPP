# Adapters

Place your adapters here. Supported today: LoRA, in two formats.

- **PEFT directory**: a folder containing `adapter_model.safetensors` + `adapter_config.json`
- **ComfyUI single file**: a `.safetensors` file with alpha baked into the tensor keys (no config needed)

Point the server to this directory:

```bash
./build/ace-server --models ./models --adapters ./adapters
```
