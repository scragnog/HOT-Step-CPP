// The tier change's guards (desk/stage.ts, shell/tier.ts, desk/layout.ts): a tier change on a full desk, clicked as a
// user does, with what the page counted in the change's own drawing (the View Transition's update, before anything
// moves: the test layer's `tier.sync.*` counters, src/app/tally.ts) held to a limit, so the costs removed cannot come
// back unseen: every tile drawn, laid out and measured at the new tier before the motion starts, when the desk has been
// laid out at that tier before; the desk measured twice in one change; the desk restyled whole to be measured. These hold
// on the rebuild only (the concept counts nothing).
//
// The limits, on every ACE-Step module, pedal and side unit at 1920x1080: a return to a tier draws the tiles in view and
// holds the rest (3 of the 10 tiles at Studio, the Pedalboard among them: 95 models asked and 7 tiles measured in the
// drawing, where a first change asks 133 and measures 10); a first change measures the desk once, with no class set on
// the desk to measure it.

const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'repetition', 'sampler', 'guidance', 'perf', 'latent', 'seed', 'takes', 'post'];
const PEDS = ['tile', 'refiner', 'ss', 'ppvae', 'lift', 'nat', 'gain', 'vst', 'master', 'norm', 'lrc', 'wh', 'art', 'q'];
const full = (tier = 'lab') => t => t.do(`a full ${tier} desk`, ([tier, mods, pedals]) => { S.tier = tier; S.mods = mods.filter(id => MODS[id]); S.pedals = pedals.filter(id => PED[id]); S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); render(); scrollTo(0, 0); }, [tier, ACE, PEDS], 600);
const reset = t => t.js(() => window.__tally(true));
const counted = t => t.js(() => window.__tally());
/* PERF_SHOW=1 prints every count (each check fails, saying it) */
const within = (t, label, got, max) => t.ok(`${label}: at most ${max}`, ([n, m]) => (n <= m ? true : `${n}`), [got ?? 0, process.env.PERF_SHOW ? -1 : max]);
const atLeast = (t, label, got, min) => t.ok(`${label}: at least ${min}`, ([n, m]) => (n >= m ? true : `${n}`), [got ?? 0, process.env.PERF_SHOW ? 1e9 : min]);
/* the desk as it stands: every tile's markup, its place and its classes included */
const desk = () => document.getElementById('bento').outerHTML;

export default [
  { id: 'a-return-to-a-tier-draws-only-what-is-in-view', run: async t => {
    await full('lab')(t);
    await t.click('#tiers [data-tier="studio"]');
    const first = await t.js(desk);
    await t.click('#tiers [data-tier="lab"]');
    await reset(t);
    await t.click('#tiers [data-tier="studio"]');
    const n = await counted(t);
    await t.eq('the tier is Studio', () => S.tier, 'studio');
    await atLeast(t, 'tiles held out of view while the change is drawn (the Pedalboard among them)', n['stage.held'], 3);
    await within(t, 'models asked in the change\'s own drawing', n['tier.sync.model'], 110);
    await within(t, 'tiles measured in the change\'s own drawing (those in view)', n['tier.sync.measure.tiles'], 7);
    await t.ok('every held tile drawn: as many drawn as held', ([h, d]) => (h === d ? true : `${d} of ${h}`), [n['stage.held'] ?? 0, n['stage.drawn'] ?? 0]);
    await t.ok('no tile is left hidden', () => [...document.querySelectorAll('#bento > *')].every(e => !e.style.contentVisibility) || 'a tile keeps content-visibility');
    await t.eq('the desk is exactly as the first change to Studio drew it', desk, first);
  } },
  { id: 'a-first-change-measures-the-desk-once-without-restyling-it-whole', run: async t => {
    await full('studio')(t);
    await t.js(() => { window.__cls = []; new MutationObserver(rs => rs.forEach(r => window.__cls.push(r.oldValue ?? ''))).observe(document.getElementById('bento'), { attributes: true, attributeOldValue: true, attributeFilter: ['class'] }); });
    await reset(t);
    await t.click('#tiers [data-tier="lab"]');
    const n = await counted(t);
    await t.eq('the tier is Lab', () => S.tier, 'lab');
    await within(t, 'times the desk measured its tiles for a first change to a tier', n.measure, 1);
    await t.eq('the desk never takes a class to be measured (a class on it restyles it whole)', () => window.__cls.filter(c => /measuring/.test(c)), []);
  } },
];
