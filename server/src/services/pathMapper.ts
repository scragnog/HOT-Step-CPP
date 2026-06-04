// pathMapper.ts — Windows-to-Docker path translation
//
// Album presets in the DB store Windows-native paths (e.g.
// "D:\Ace-Step-Latest\All LoKR Files\sidestep\xl-base-turbo-05\file.safetensors").
// These don't exist inside the Docker container. This module translates them
// to container mount points using a prefix map from DOCKER_PATH_MAP.
//
// When DOCKER_PATH_MAP is not set (Windows-native mode), all functions are no-ops.
//
// Configuration (in .env.docker):
//   DOCKER_PATH_MAP={"D:\\Ace-Step-Latest\\All LoKR Files\\sidestep\\xl-base-turbo-05":"/app/adapters","D:\\Ace-Step-Latest\\Datasets-LoRA-LoKR":"/app/datasets"}

interface PathMapping {
  /** Windows prefix (normalized: forward slashes, lowercase, no trailing slash) */
  winPrefix: string;
  /** Container mount point (no trailing slash) */
  mountPoint: string;
}

let mappings: PathMapping[] = [];
let initialized = false;

/** Normalize a path for comparison: forward slashes, lowercase, no trailing slash */
function normalize(p: string): string {
  return p.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
}

/** Parse DOCKER_PATH_MAP from environment. Called once on first use. */
function init(): void {
  if (initialized) return;
  initialized = true;

  const raw = process.env.DOCKER_PATH_MAP;
  if (!raw) return;

  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) {
      console.error('[PathMapper] DOCKER_PATH_MAP is not a JSON object, ignoring');
      return;
    }

    for (const [winPath, mountPath] of Object.entries(parsed)) {
      if (typeof mountPath !== 'string') continue;
      mappings.push({
        winPrefix: normalize(winPath),
        mountPoint: (mountPath as string).replace(/\/+$/, ''),
      });
    }

    // Sort by prefix length descending so longer (more specific) matches win
    mappings.sort((a, b) => b.winPrefix.length - a.winPrefix.length);

    if (mappings.length > 0) {
      console.log(`[PathMapper] Loaded ${mappings.length} path mapping(s):`);
      for (const m of mappings) {
        console.log(`[PathMapper]   ${m.winPrefix} → ${m.mountPoint}`);
      }
    }
  } catch (err) {
    console.error(`[PathMapper] Failed to parse DOCKER_PATH_MAP: ${err}`);
  }
}

/**
 * Translate a path if it matches a known Windows prefix.
 * Returns the original path unchanged if no mapping matches or if
 * DOCKER_PATH_MAP is not configured (Windows-native mode).
 */
export function mapPath(inputPath: string | undefined): string | undefined {
  if (!inputPath) return inputPath;
  init();
  if (mappings.length === 0) return inputPath;

  const norm = normalize(inputPath);
  for (const m of mappings) {
    if (norm.startsWith(m.winPrefix)) {
      // Replace prefix, preserve the rest of the path
      const remainder = inputPath
        .replace(/\\/g, '/')
        .substring(m.winPrefix.length)
        .replace(/^\//, '');  // remove leading slash if present
      const mapped = remainder ? `${m.mountPoint}/${remainder}` : m.mountPoint;
      return mapped;
    }
  }

  return inputPath;
}

/**
 * Check if path mapping is active (DOCKER_PATH_MAP is configured).
 */
export function isPathMappingActive(): boolean {
  init();
  return mappings.length > 0;
}
