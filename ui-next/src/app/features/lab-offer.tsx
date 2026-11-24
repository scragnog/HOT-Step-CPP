// The Lab offer (the concept's labOfferHTML, 30-desk.js): a Lab module added outside Lab from a card says so in the
// popover by the card, with "Switch to Lab and add it" and "Not now" (adding.ts offerLab opens it).

import { labOffer } from '../../derive.ts';
import { defOf } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { labAdd } from './adding.ts';

export function LabOffer({ id }: { id: string }) {
  const say = useSay(), d = defOf(id), desk = useDesk(s => s.desk);
  if (!d) return null;
  const o = labOffer(d, desk);
  return (
    <>
      <h4 className="pop-title">{say(ui('labOffer.title', { name: o.name }))}</h4>
      <p className="pop-p">{o.parked ? say(ui('pop.labParked', o.parked)) : ''}{say(ui('labOffer.body'))}</p>
      <div className="pop-row">
        <button type="button" className="small-btn primary" data-labadd={id} onClick={() => labAdd(id)}>{say(ui('labOffer.go'))}</button>
        <button type="button" className="small-btn" data-closepop="">{say(ui('labOffer.no'))}</button>
      </div>
    </>
  );
}
