/** Keep a dataset's trigger unless another dataset already owns it. */
export function uniqueDatasetTrigger(base: string, slug: string, taken: ReadonlySet<string>): string {
  if (!taken.has(base.toLowerCase())) return base;
  const suffix = slug.startsWith(`${base}_`) || slug.startsWith(`${base}-`)
    ? slug.slice(base.length + 1) : slug;
  const root = `${base}_${suffix}`;
  let candidate = root;
  for (let n = 2; taken.has(candidate.toLowerCase()); n++) candidate = `${root}_${n}`;
  return candidate;
}
