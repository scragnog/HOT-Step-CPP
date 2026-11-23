// Imported first by ui/src/main.tsx. Zod's JIT probes `new Function('')` when a
// schema is built, which the page's CSP (no 'unsafe-eval') refuses and reports.
// A side-effect module so it runs before any contract module builds a schema;
// it lives here so it configures the same zod copy the contracts resolve.
import { config } from 'zod/v4';

config({ jitless: true });
