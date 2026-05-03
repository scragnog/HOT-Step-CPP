# Model Manager — In-App Model Download & Management

## Problem
HOT-Step-CPP supports GGUF models from multiple HuggingFace repos but provides no way
for users to discover, download, or manage them from within the app. Users must manually
find repos, download files, and place them in the correct directory.

There are ~104 GGUF files across 5 repos spanning 5 pipeline roles (DiT, LM, Embedding,
VAE, PP-VAE), multiple model scales (Standard 2B, XL 4B), variants (turbo, sft, base,
merges), and quantisation levels (BF16, Q8, Q6, Q5, Q4, MXFP4). This is overwhelming
without guidance.

## Solution
An in-app **Model Manager** accessible from the Models dropdown in the global param bar.
Combines **curated starter packs** for newcomers with a **full browsable catalogue** for
power users. Shows what's already installed, supports concurrent resumable downloads.

---

## Registry & Data Model

### Model Registry
A curated JSON file shipped with the app: `server/src/data/model-registry.json`

Each entry contains:
- `id` — unique identifier (e.g. `dit-xl-turbo-q8`)
- `filename` — exact GGUF filename
- `role` — pipeline role: `dit` | `lm` | `embedding` | `vae` | `pp-vae`
- `displayName` — human-friendly name
- `scale` — `standard` | `xl` (DiT only)
- `variant` — model variant (turbo, sft, base, merge, etc.)
- `quant` — quantisation level
- `sizeBytes` — file size for download estimation
- `repo` — HuggingFace repo (org/name)
- `description` — what makes this model/variant special
- `tags` — for filtering (e.g. `recommended`, `blackwell`, `experimental`)
- `recommendedSettings` — optional inference parameter suggestions

### Starter Packs
All packs include ScragVAE (BF16) and PP-VAE.

| Pack | DiT | LM | Embedding | VAE | ScragVAE | PP-VAE | Total |
|------|-----|----|-----------|-----|----------|--------|------:|
| **Quick Start** | Turbo Q8 (2.4GB) | 4B Q8 (4.2GB) | Q8 (748MB) | BF16 (322MB) | BF16 (322MB) | F32 (644MB) | ~8.6 GB |
| **Minimal** | Turbo Q4 (1.4GB) | 1.7B Q8 (1.9GB) | Q8 (748MB) | BF16 (322MB) | BF16 (322MB) | BF16 (322MB) | ~5 GB |
| **XL Quality** | XL Turbo Q8 (5GB) | 4B Q8 (4.2GB) | Q8 (748MB) | BF16 (322MB) | BF16 (322MB) | F32 (644MB) | ~11.2 GB |
| **Blackwell** | XL Turbo MXFP4 (2.5GB) | 4B Q8 (4.2GB) | Q8 (748MB) | BF16 (322MB) | BF16 (322MB) | F32 (644MB) | ~8.7 GB |

Packs are "smart" — they detect which files are already installed and only offer to
download the missing ones.

### HuggingFace Repos
| Repo | Content |
|------|---------|
| `Serveurperso/ACE-Step-1.5-GGUF` | Official DiT (Standard + XL), LM, Embedding, VAE |
| `scragnog/ace-step-1.5-gguf-merge-models` | Custom XL DiT merges (task arithmetic) |
| `scragnog/Ace-Step-1.5-MXFP4-Quants` | MXFP4 quantised DiT (Blackwell optimised) |
| `scragnog/HOT-Step-CPP-PP-VAE` | Post-processing VAE (spectral cleanup) |
| `scragnog/Ace-Step-1.5-ScragVAE` | Improved VAE decoder (better HF content) |

---

## Server-Side Architecture

### Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/model-manager/registry` | Registry + `installed` status per file |
| `POST` | `/api/model-manager/download` | Start download (body: `{ fileId }`) → job ID |
| `GET` | `/api/model-manager/downloads` | SSE stream of all active download progress |
| `POST` | `/api/model-manager/download/:jobId/cancel` | Cancel active download |
| `POST` | `/api/model-manager/download/:jobId/resume` | Resume paused/failed download |
| `DELETE` | `/api/model-manager/files/:filename` | Delete model file from disk |

### Download Service (`modelDownloadService.ts`)
- Downloads from `https://huggingface.co/{repo}/resolve/main/{filename}`
- HTTP `Range` headers for resumption — writes `{filename}.part`, renames on completion
- Tracks concurrent downloads in `Map<jobId, DownloadJob>`
- Each job tracks: bytes downloaded, total bytes, speed (bytes/sec), ETA
- SSE endpoint streams all jobs to UI every ~500ms
- Network failures → job enters `paused` state, user can resume
- Multiple concurrent downloads supported

### Files
```
server/src/services/modelDownloadService.ts   — download engine
server/src/routes/modelManager.ts             — API routes
server/src/data/model-registry.json           — curated catalogue
```

---

## Frontend UI

### Entry Point
A **"Get More Models"** button at the bottom of the `ModelsDropdown` component in the
global param bar. Opens a full-screen modal.

### Modal Layout: `ModelManagerModal`

**Active Downloads Banner** (top, only visible when downloads active)
- Compact progress bars per download: filename, %, speed, ETA
- Cancel button per download

**Starter Packs** (4 horizontal cards)
- Pack name + total download size
- List of included models: ✅ (installed) or ⬇️ (will download)
- "Download Missing" button (disabled if all installed)
- Shows actual remaining download size

**Full Catalogue** (tabbed by role)
- Tabs: DiT | LM | Text Encoder | VAE | PP-VAE
- DiT sub-groups: Standard (2B) → XL (4B) → XL Merges → MXFP4
- Each model row: display name, quant badge, description, size, status, actions
- Expandable info sections explaining variant differences
- Delete button with confirmation for installed models

**Footer**
- Models directory path + total disk usage
- "Open Models Folder" button

### Component Structure
```
ui/src/components/model-manager/
  ModelManagerModal.tsx        — modal shell
  StarterPackCard.tsx          — pack card
  ModelCatalogueTab.tsx        — tabbed catalogue
  ModelRow.tsx                 — single model row
  DownloadProgressBar.tsx      — reusable progress bar
  useModelRegistry.ts          — hook: registry + installed state
  useDownloadStream.ts         — hook: SSE for download progress
```

### Colour Language
- **Pink** accent (matching Models section in global bar)
- **Emerald** pills for installed models
- **Sky** progress bars for active downloads
- **Amber** for paused/failed downloads

---

## API Integration
Frontend API client additions in `api.ts`:
```typescript
export const modelManagerApi = {
  registry: () => get<ModelRegistry>('/model-manager/registry'),
  download: (fileId: string) => post<{ jobId: string }>('/model-manager/download', { fileId }),
  cancel: (jobId: string) => post('/model-manager/download/' + jobId + '/cancel'),
  resume: (jobId: string) => post('/model-manager/download/' + jobId + '/resume'),
  deleteFile: (filename: string) => del('/model-manager/files/' + filename),
  // SSE: new EventSource('/api/model-manager/downloads')
};
```
