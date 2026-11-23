// The containers the slice uses. Each holds its children, already drawn by the renderer, in the housing of the
// concept's markup: row is the .knobs row, rack the .pextra.rackrow strip, labelled the lcdsel pairing (the control,
// then its caps label). A container none of whose children is drawn is not drawn at all (the renderer asks the core,
// nodeShown).

import { nodeWord } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import type { ViewProps } from './types.ts';

export function Row({ children }: ViewProps) {
  return <div className="knobs">{children}</div>;
}

export function Rack({ children }: ViewProps) {
  return <div className="pextra rackrow">{children}</div>;
}

export function Labelled({ node, nodeKey, children }: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  return <div className="lcdsel">{children}<span className="kl">{say(nodeWord(ix, node, desk, nodeKey, 'label'))}</span></div>;
}
