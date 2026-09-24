// pluginTypes.ts — Type definitions for the Lua plugin system
//
// These types mirror the JSON schema served by GET /api/plugins
// (which proxies to ace-server's GET /plugins endpoint).

/** Plugin parameter schema from Lua plugin metadata */
export interface PluginParamSchema {
  key: string;
  type: 'slider' | 'select' | 'toggle' | 'text';
  /** text only: render a multi-line box (lead sheets, anything pasted). */
  multiline?: boolean;
  label: string;
  hint?: string;
  transform?: string;
  // slider
  default?: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  // select
  options?: { value: string; label: string }[];
  // conditional visibility
  visible_when?: { key: string; equals: string };
}

/** Plugin metadata from Lua plugin files */
export interface PluginInfo {
  name: string;
  display: string;
  description?: string;
  accent?: string;
  // solver-specific
  nfe?: number;
  order?: number;
  needs_model?: boolean;
  stateful?: boolean;
  stochastic?: boolean;
  /** Full-loop solver: defines sample() and drives its own iteration instead of
   *  a per-step step(). The engine has always served this field
   *  (lua-plugin-registry.h:303); it was simply never typed here. Backends
   *  other than ACE cannot run these — see SamplerPluginControls.tsx. */
  owns_loop?: boolean;
  params: PluginParamSchema[];
}

/** Full plugin registry from GET /api/plugins */
export interface PluginRegistry {
  solvers: PluginInfo[];
  schedulers: PluginInfo[];
  guidance: PluginInfo[];
  postprocess?: PluginInfo[];
}
