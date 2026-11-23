// StudioApp.tsx — the studio shell. A stub until slice 2 brings the layout:
// the shared runtime plus one live readout, so a mid-render switch is visible.

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '../../context/AuthContext';
import { GlobalParamsProvider } from '../../context/GlobalParamsContext';
import { useAppRuntime } from '../../app/useAppRuntime';
import { RestartingOverlay, ServerOfflineOverlay } from '../../app/ServerOverlays';
import { switchShell } from '../../app/shell';
import { songsOptions } from '../../queries/shell-queue/songs';
import { useAudioGenQueueSelector } from '../../stores/audioGenQueueStore';
import { Button } from '../../components/shared/kit/Button';
import { SectionPanel } from '../../components/shared/kit/Panels';
import { SkinPicker } from '../../components/shared/kit/SkinPicker';
import { useSkin } from '../../themes/registry';

const StudioContent: React.FC = () => {
  const { token } = useAuth();
  const { activeJobCount, isOffline, isRestarting } = useAppRuntime();
  const queueLength = useAudioGenQueueSelector(s => s.items.length);
  const { data } = useQuery({ ...songsOptions(), enabled: !!token });
  const latest = data?.songs[0];
  const { skin, setSkin } = useSkin();

  if (isRestarting) return <RestartingOverlay />;
  if (isOffline) return <ServerOfflineOverlay />;

  return (
    <div className="p-6 max-w-xl">
      <SectionPanel
        title="Studio shell (preview)"
        actions={<Button size="sm" variant="secondary" onClick={() => switchShell('classic')}>Back to classic</Button>}
        bodyClassName="p-4 flex flex-col gap-2"
      >
        <p>This is the empty studio shell. It runs the same app runtime as classic.</p>
        <p data-testid="studio-queue">Queue: {queueLength} item{queueLength === 1 ? '' : 's'}, {activeJobCount} running</p>
        <p data-testid="studio-latest">Latest song: {latest?.title ?? 'none yet'}</p>
        <div className="flex items-center gap-2 text-xs text-ink-muted">Skin <SkinPicker skin={skin} onChange={setSkin} /></div>
      </SectionPanel>
    </div>
  );
};

const StudioApp: React.FC = () => (
  <GlobalParamsProvider>
    <StudioContent />
  </GlobalParamsProvider>
);

export default StudioApp;
