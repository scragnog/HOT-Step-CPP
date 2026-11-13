# RunPod Pod template

HOT-Step runs on a RunPod GPU Pod as one container. The Node server serves the UI
on port 3001 and starts the C++ engine internally on port 8085. The `runpod` Docker
target includes the app, engine binaries and built-in Lua plugins. Model weights
are downloaded after deployment. This guide covers GPU Pods, not Serverless workers.

> **No authentication:** this Pod has no app login. RunPod's HTTP proxy makes port
> 3001 reachable at `https://<pod-id>-3001.proxy.runpod.net`. Anyone who can reach
> that address can access the library and generation API. Put an authenticated
> access layer in front of the app before exposing sensitive data or sharing access.
> Enabling the HTTP port alone does not add authentication.

## Pull the published image

The separate `RunPod image` workflow (`.github/workflows/runpod-image.yml`)
publishes `ghcr.io/scragnog/hot-step-cpp:vX.Y.Z` when a GitHub release is
published (a tag push alone does not), or when it is run by hand on a `v*` tag.
`:latest` moves only for a non-prerelease tag without a hyphen. A failed run
publishes nothing for that tag, so wait for the workflow to succeed before
pulling. Replace `vX.Y.Z` with the release tag you want:

```sh
docker pull ghcr.io/scragnog/hot-step-cpp:vX.Y.Z
```

Use that image in the RunPod template below, or `:latest` for the most recent
stable release image. Prefer a version tag for repeatable deployments. The
published image uses the portable CUDA architectures and build arguments below.
Models are downloaded after deployment, and data persists under `/workspace`.

## Build and publish your image

To build your own image, use a Linux x86_64 Docker builder with enough disk
space for the CUDA build layers. On Apple Silicon, use a remote x86_64 builder;
building CUDA under emulation is slow. The recursive clone must initialize
`engine/ggml` and `engine/vendor/vst3sdk` before the build.

Replace `YOUR_REGISTRY` and `YOUR_VERSION` below with your registry namespace and
an immutable version tag, such as the source commit hash.

```sh
git clone --recursive https://github.com/scragnog/HOT-Step-CPP.git
cd HOT-Step-CPP
docker buildx build --platform linux/amd64 --target runpod \
  --build-arg 'CUDA_ARCHS=75-virtual;80-virtual;86-real;89-real;90-real;120a-real' \
  --build-arg ENABLE_TRT=0 \
  --build-arg BUILD_JOBS=2 \
  -t YOUR_REGISTRY/hot-step-cpp:YOUR_VERSION --load .
```

`CUDA_ARCHS` selects the GPU architectures supported by the image. Match it to
the GPU you will deploy on. `ENABLE_TRT=0` disables the optional native TensorRT
path and its packages; GGML CUDA remains enabled. `BUILD_JOBS=2` limits parallel
compilation to reduce build memory use. Model files are not baked into the image.

For a local smoke test on a Docker host with NVIDIA Container Toolkit, start the
container in one terminal:

```sh
docker run --rm --gpus all -p 127.0.0.1:3001:3001 \
  -v hot-step-workspace:/workspace \
  YOUR_REGISTRY/hot-step-cpp:YOUR_VERSION
```

In another terminal, check startup and open `http://localhost:3001`:

```sh
curl -f http://localhost:3001/api/health
```

On a fresh, empty `/workspace` there are no models yet, so expect health to return
200 and the UI to load with `aceServer.status=disconnected`, `engine.ready=false`
and `engine.noModels=true`, and a banner pointing at the Model Manager. The engine
logs `ERROR: no models found in /workspace/models` and the server logs `No models
installed`; that is normal until models are installed, and it does not count as an
engine crash. After testing, authenticate to your registry and publish the same tag:

```sh
docker login YOUR_REGISTRY
docker push YOUR_REGISTRY/hot-step-cpp:YOUR_VERSION
```

## Create the RunPod template

Create a template in the RunPod console using your pushed image. See RunPod's
[custom template guide](https://docs.runpod.io/pods/templates/create-custom-template)
for the console workflow.

| Setting | Value |
|---|---|
| Compute type | NVIDIA GPU supported by your `CUDA_ARCHS` build |
| Container image | `YOUR_REGISTRY/hot-step-cpp:YOUR_VERSION` |
| Container start command | Leave blank to use the image entrypoint |
| HTTP ports | `3001` |
| TCP ports | None required by the app |
| Volume mount path | `/workspace` |

Allocate container disk for the image and a persistent volume large enough for
your model packs, generated audio and training data. A RunPod
[network volume](https://docs.runpod.io/storage/network-volumes) can retain data
when replacing a Pod; check availability in the GPU's data center. For a private
image, configure registry credentials in RunPod before deployment.

Enable HTTP port 3001 to reach the app through RunPod's proxy. Re-read the access
warning above before enabling it. Do not expose engine port 8085; the Node server
reaches the engine inside the container.

## Persistent layout and environment

The `runpod` target sets `HOT_STEP_DATA_ROOT=/workspace`. The entrypoint uses this
guard to create persistent directories and link the app's container paths into
them. The regular local Compose target keeps its own mounts.

| Volume path | Container path | Content |
|---|---|---|
| `/workspace/models` | `/app/models` | GGUF and optional ONNX models |
| `/workspace/adapters` | `/app/adapters` | LoRA and LoKr adapters |
| `/workspace/data` | `/app/server/data` | SQLite database, generated audio, training data and app state |
| `/workspace/logs` | `/app/logs` | Per-session server, engine, generation and training logs |

The target also sets `ACESTEPCPP_EXE=/app/engine/ace-server`,
`ACESTEPCPP_MODELS=/workspace/models`, `ACESTEPCPP_ADAPTERS=/workspace/adapters`
and `DATA_DIR=/workspace/data`, and turns on GGML's CUDA graph optimization with
`GGML_CUDA_GRAPH_OPT=1`. If you change the mount path, update `HOT_STEP_DATA_ROOT`
and the models, adapters and data paths together. Do not set `HOT_STEP_ROOT` to enable portable-release mode in this
container; its entrypoint uses the container layout above. See
[Configuration](config.md#other-env-files) for the local Compose environment file.

The server binds `127.0.0.1` by default, so a pod must opt in. Set `SERVER_HOST=0.0.0.0` and
`SERVER_PORT=3001` to expose the Node server to the proxy. The proxy sends a `Host` header the
server does not know, so also set one of these:

- `CORS_ORIGINS=https://<pod-id>-3001.proxy.runpod.net` if you use the UI through the proxy.
  This allows the host and lets the browser make write requests (POST, PUT, DELETE).
- `ALLOWED_HOSTS=<pod-id>-3001.proxy.runpod.net` if you only need reads or non-browser clients.
  Browser writes from that origin still get 403 without `CORS_ORIGINS`.

Without either, every request through the proxy gets 403.
`ACESTEPCPP_HOST=127.0.0.1` and `ACESTEPCPP_PORT=8085` keep the engine inside the
container. The entrypoint refuses to replace a non-empty directory or a mounted
directory at one of the container paths above; mount the volume at `/workspace`
rather than mounting those paths separately.

Open the Model Manager and download a compatible pack before generating music.
Existing model files can be placed in `/workspace/models` before startup. For
startup failures, inspect the Pod's container logs and `/workspace/logs`. Back up
the persistent volume before deleting or replacing storage.

The engine starts by itself once models are installed: when the Model Manager's
download queue drains, or, for files copied in by hand, when a `.gguf` that has
stopped growing appears in `/workspace/models` (checked every 5 seconds). No Pod
restart is needed. Then check `/api/health` again: `aceServer.status` should no
longer be `disconnected` and `engine.ready` should be true. Inspect
`engine.bootStatus` and the container logs if not; an HTTP response alone does not
prove the engine started. The image's Docker `HEALTHCHECK` requests `/api/health`
every 30 seconds after a 120-second start period, so it reports healthy whenever the
Node server answers, with or without a running engine.
