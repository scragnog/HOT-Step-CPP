# Windows archive smoke test

Run from the checkout on Windows with Node 20-24:

```powershell
node release/smoke-artifact.mjs release/out/HOT-Step-CPP-vVERSION-win-x64-cpu.zip
node --test release/smoke-artifact.test.mjs
```

The optional second argument is an existing output parent outside the checkout.
The third argument is the server and engine timeout in seconds, default 60,
maximum 600. Extraction has a separate 120-second timeout. Every invocation
creates a unique `hotstep-artifact-*` directory under the output parent, with
the archive extracted directly into `HOT Step smoke`. The helper validates both
flat Windows packages and versioned packages. In a flat package, `runtime/node.exe`,
`server/server.mjs` and `engine/ace-server.exe` sit at the root. In a protocol-1
package they sit under the `app/<id>/` payload named by `current.json`. See
[Windows updater layout](../docs/dev/releasing.md#windows-updater-layout-protocol-1).
Other scripts can use the same resolver with
`node release/smoke-artifact.mjs payload-root <extracted-root>`.

The gate runs the checkout's existing `smoke-boot.mjs` using the extracted Node
runtime against the extracted server bundle. It uses temporary data and profile
directories, a deliberately absent engine and a CPU marker so boot does not
download CUDA runtimes. The engine is then launched separately with four
header-only GGUF stubs and a free localhost port. It must serve `/health`,
`/props`, `/plugins` and `/props` again, then remain alive for two seconds.
Requests have two-second deadlines. These checks do not load models or test
GPU inference.

The server and engine run with only Windows system directories on `PATH`, so
toolkit or build-tree directories cannot supply missing packaged DLLs. The
engine still uses DLLs next to its executable and the Windows system loader.
The current process environment is otherwise inherited; package-local config
is part of what is tested. Each command is awaited and reaped. On timeout or
interruption, Windows `taskkill /PID <owned-pid> /T /F` kills its process tree.
No dev stack, launcher or GPU job is started.

## Backend check

After the endpoint smoke, the gate runs `ace-server --backend-check`. The
engine loads its GGML backends as it would for a generation, without models,
and prints a JSON report: each registered backend with its module and devices,
the backend it would select, a CPU backend init, and each `ggml-*.dll` beside
the exe. For a DLL that did not register, the report gives the Windows load
error and follows the import chain to the first DLL that will not load. A
Vulkan backend that loads but returns no registration also gets a direct
`vkCreateInstance` result. Release GGML loads backends silently, so without
this a missing dependency looks the same as a machine with no GPU.

`engine/.variant` sets the expectation, defaulting to `cuda` as the app does.
The check fails unless the CPU backend initialises and a backend is selected.
The expected GPU backend (`ggml-cuda.dll` or `ggml-vulkan.dll`) must register,
with any number of devices, or fail for an identified missing driver:
`nvcuda.dll` or `vulkan-1.dll` failing with error 126 and not on the loader's
search path, or `VK_ERROR_INCOMPATIBLE_DRIVER`. A driver DLL that exists but
fails (127, 193, 1114 and so on) is a failure, not an absent driver. Anything
else fails: a missing package DLL, a missing runtime DLL, a missing export, a
bare error 126 with no missing import found, or a crash. A CPU variant DLL that
fails to load also fails the check.

CUDA archives do not ship the CUDA runtime. The gate provisions it the way the
app does: `engine/.cuda-version` selects the registry's `cuda-runtime` or
`cuda12-runtime` pack, each file is downloaded from Hugging Face at its
`repo`/`repoPath` and checked against `sizeBytes`, and the folder goes first on
`PATH`. Pass a cache folder as the fourth argument to reuse downloads; files of
the right size are not fetched again. Without one, the files go under the
evidence directory's isolated `data/tools/runtime`.

Negative checks then run on the same extraction, and each must fail: the GPU
backend DLL removed (every `ggml-cpu*.dll` for the CPU archive), and for CUDA a
copy of the runtime folder without its largest DLL (cuBLASLt). The cache folder
itself is never modified. A negative check that passes fails the gate.

### Runtime cache for CI

The pack is about 533 MB for CUDA 13 and 807 MB for CUDA 12. Key the cache on
the registry entries, so model edits elsewhere in the registry keep it warm:

```yaml
- name: CUDA runtime cache key
  if: startsWith(matrix.variant, 'cuda')
  id: cuda-runtime
  shell: pwsh
  run: |
    $major = '${{ matrix.cuda_version }}'.Split('.')[0]
    "major=$major" >> $env:GITHUB_OUTPUT
    "key=$(node release/smoke-artifact.mjs runtime-key $major)" >> $env:GITHUB_OUTPUT
- name: CUDA runtime cache
  if: startsWith(matrix.variant, 'cuda')
  uses: actions/cache@v4
  with:
    path: ${{ runner.temp }}/cuda-runtime-${{ steps.cuda-runtime.outputs.major }}
    key: ${{ steps.cuda-runtime.outputs.key }}
```

Then pass `${{ runner.temp }}/cuda-runtime-<major>` as the fourth argument.
`node release/smoke-artifact.mjs runtime-fetch <major> <dir>` fills a folder
without running the gate and prints the bytes and seconds taken.

## Evidence

The printed evidence directory is retained on success and failure. It contains
`result.json`, `extract.log`, `server.log`, and, once the server passes,
`engine.log`, `backend.log` and one `backend-negative-*.log` per negative
check, plus the extraction and isolated test state. Upload that directory
as CI failure evidence, or inspect it locally and delete it when finished.
`server.log` includes the existing smoke helper's last 40 server output lines
on failure. Exit 0 means every check passed; any extraction, boot, HTTP, backend,
download or cleanup failure returns nonzero. Workflow wiring and the owning developer docs
are handled by the release workflow slice.
