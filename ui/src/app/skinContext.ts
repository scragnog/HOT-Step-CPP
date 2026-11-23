// skinContext.ts — how the host's runtime reaches skin-api. A skin never reads this directly.

export interface SkinRuntime {
  shellId: string;
  /** Start a Create render from content fields. Merges the global controls and settings flags. */
  generate: (content: Record<string, unknown>) => void;
}

/** The runtime of the mounted shell, for plain functions such as generate(). Set by ShellHost. */
export let activeRuntime: SkinRuntime | null = null;
export const setActiveRuntime = (rt: SkinRuntime | null): void => { activeRuntime = rt; };
