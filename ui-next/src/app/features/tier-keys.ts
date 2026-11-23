// Alt+1, 2 and 3 choose Quick, Studio and Lab (31-shell.js).

import { onKey } from '../register/keys.ts';
import { setTierFromView } from '../shell/tier.ts';

const TIERS = ['quick', 'studio', 'lab'] as const;

onKey({
  id: 'tier-keys',
  when: e => e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey && /^Digit[123]$/.test(e.code),
  run: e => { e.preventDefault(); setTierFromView(TIERS[+e.code.slice(5) - 1]!); },
});
