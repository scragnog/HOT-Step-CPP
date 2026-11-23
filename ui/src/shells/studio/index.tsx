// studio/index.tsx — the studio shell. A stub until slice 2 brings the layout: one live
// readout of the queue and the latest song, built only on skin-api and the kit.

import React from 'react';
import { openClassic, useQueue, useSongs, useThemeSkin } from '../../skin-api';
import { Button } from '../../components/shared/kit/Button';
import { SectionPanel } from '../../components/shared/kit/Panels';
import { SkinPicker } from '../../components/shared/kit/SkinPicker';

const StudioApp: React.FC = () => {
  const { items, active } = useQueue();
  const { songs } = useSongs();
  const { skin, setSkin } = useThemeSkin();

  return (
    <div className="p-6 max-w-xl">
      <SectionPanel
        title="Studio shell (preview)"
        actions={<Button size="sm" variant="secondary" onClick={openClassic}>Back to classic</Button>}
        bodyClassName="p-4 flex flex-col gap-2"
      >
        <p>This is the empty studio shell. It runs the same app runtime as classic.</p>
        <p data-testid="studio-queue">Queue: {items.length} item{items.length === 1 ? '' : 's'}, {active} running</p>
        <p data-testid="studio-latest">Latest song: {songs[0]?.title ?? 'none yet'}</p>
        <div className="flex items-center gap-2 text-xs text-ink-muted">Skin <SkinPicker skin={skin} onChange={setSkin} /></div>
      </SectionPanel>
    </div>
  );
};

export default StudioApp;
