// The containers. Each holds its children, already drawn by the renderer, in the housing of the concept's markup: row
// is the .knobs row, rack the .pextra.rackrow strip, well the recessed strip under an instrument (the Sampler's), lcd an
// LCD panel, foot a tile's foot row. labelled pairs a caps label with one small control, as the surface it stands on
// pairs them: on a tile, a foot or a well the label comes first (the concept's foot-ctl); on hardware (a pedal, a side
// unit, an LCD) the control comes first and the caps label under it (lcdsel). A container none of whose children is
// drawn is not drawn at all (the core's nodeShown).

import { definePart, useSay, type PartView } from '../kit/index.ts';

export const Row = ({ children }: PartView) => <div className="knobs">{children}</div>;
export const Rack = ({ children }: PartView) => <div className="pextra rackrow">{children}</div>;
export const Well = ({ children }: PartView) => <div className="knobs samp-strip">{children}</div>;
export const Lcd = ({ children }: PartView) => <div className="lcd">{children}</div>;
export const Foot = ({ children }: PartView) => <div className="tile-foot">{children}</div>;

export function Labelled(v: PartView) {
  const say = useSay(), hardware = v.surface === 'pedal' || v.surface === 'unit' || v.surface === 'lcd';
  return hardware
    ? <div className="lcdsel">{v.children}<span className="kl">{say(v.label)}</span></div>
    : <span className="foot-ctl"><span className="kl">{say(v.label)}</span>{v.children}</span>;
}

definePart('row', Row);
definePart('rack', Rack);
definePart('well', Well);
definePart('lcd', Lcd);
definePart('foot', Foot);
definePart('labelled', Labelled);
