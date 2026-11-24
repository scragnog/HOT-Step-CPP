// The server boundary's registry (services/index.ts): each service is registered once, by name, with the interface the
// Services map gives that name, and asked for by name. main.tsx registers every one before the page loads (the
// simulated server, services/mock/index.ts; package 10's real adapters in its place). A consumer asks with service(name)
// and never imports an implementation.

import type { Services } from '../services/index.ts';

const SERVICES = new Map<keyof Services, unknown>();

export function defineService<K extends keyof Services>(name: K, impl: Services[K]): Services[K] {
  if (SERVICES.has(name)) throw new Error(`the service "${name}" is defined twice`);
  SERVICES.set(name, impl); return impl;
}
export function service<K extends keyof Services>(name: K): Services[K] {
  const s = SERVICES.get(name); if (!s) throw new Error(`no service "${name}": main.tsx registers every service before the page loads`);
  return s as Services[K];
}
