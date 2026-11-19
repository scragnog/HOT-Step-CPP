// Pure map-merge rule used by applyProfileData (paramProfiles.ts). Kept free of
// store imports so it can be unit-tested in node.

const isMap = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/**
 * backendParams is always merged over the live map: a profile carries only the
 * knobs it names, and a snapshot from another backend must not wipe this
 * backend's model paths etc. Preset files may set _mergeMaps so their
 * pluginParams merge too. Every other field (and a non-map value) is returned
 * as given.
 */
export function mergeProfileField(field: string, value: unknown, liveValue: unknown, mergeMaps: boolean): unknown {
  if ((field === 'backendParams' || (field === 'pluginParams' && mergeMaps)) && isMap(value) && isMap(liveValue)) {
    return { ...liveValue, ...value };
  }
  return value;
}
