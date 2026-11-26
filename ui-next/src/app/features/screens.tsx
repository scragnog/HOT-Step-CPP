// The other pages (the concept's 31-screens.js, RC-01): the top bar's Library, Studios, Training and Settings show the
// stand-in that says the page keeps its own place, with Back to Create where the eye is (F-35); a link that names a page
// (the brand, Latest's "Open the Library": data-goto) does what that page's button does, and focus goes to the button
// when the link that was pressed is hidden with the page it was on (F-34, C-35).

import { createPortal } from 'react-dom';
import { fill } from '../register/slots.ts';
import { ui, useSay } from '../i18n/index.ts';
import { pageWord, showScreen, useScreen } from '../shell/screens.ts';

export function Other() {
  const say = useSay(), name = useScreen(s => s.name), el = document.getElementById('other');
  if (!el || name === 'create') return null;
  const page = say(ui(pageWord(name)));
  return createPortal(<><h2>{page}</h2><p>{say(ui('other.says', { name: page }))}</p><button type="button" className="small-btn" data-goto="create">{say(ui('other.back'))}</button></>, el);
}
fill('overlays', Other);

document.addEventListener('click', e => {
  const b = (e.target as Element).closest?.<HTMLElement>('[data-goto], .top [data-nav]'); if (!b) return;
  const name = b.dataset.goto ?? b.dataset.nav!;
  showScreen(name);
  const nb = document.querySelector<HTMLElement>(`.top [data-nav="${CSS.escape(name)}"]`);
  if (nb && b !== nb && !b.getClientRects().length) nb.focus({ preventScroll: true });
});
