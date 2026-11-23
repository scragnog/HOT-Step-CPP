// What a work package asks the server for beyond the page's Engine (services/engine.ts): the build session, the
// planner's runs, a model's figures. A package defines its service in a file of its own (the mock beside it, the real
// adapter in package 10) and registers it here; main.tsx chooses mock or real for every service at once.

const SERVICES = new Map<string, unknown>();

export function defineService<T>(name: string, impl: T): T {
  if (SERVICES.has(name)) throw new Error(`the service "${name}" is defined twice`);
  SERVICES.set(name, impl); return impl;
}
export function service<T>(name: string): T {
  const s = SERVICES.get(name); if (!s) throw new Error(`no service "${name}": its package registers it`);
  return s as T;
}
