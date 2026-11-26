export default [
  { id: 'smoke', run: async t => {
    await t.eq('boots on Quick with ACE-Step', () => [S.tier, S.backend, S.songs.length], ['quick', 'ace', 6]);
    await t.click('#tiers [data-tier="studio"]');
    await t.eq('studio tier', () => S.tier, 'studio');
    await t.do('advance', () => {}, null, 5000);
    await t.reload();
    await t.eq('tier survives a reload', () => S.tier, 'studio');
    await t.eq('clock is fake', () => new Date().getUTCFullYear(), 2031);
  } },
];
