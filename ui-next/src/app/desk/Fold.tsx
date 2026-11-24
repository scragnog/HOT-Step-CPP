// The fold's button at the desk's foot (the concept's .moretiles, 30-desk.js): it names the modules Quick has folded
// away and shows them, or folds them again. Its words and which modules it holds are the core's (deskModel's fold);
// whether it stands open is the session's (store/fold.ts). A click is no step: nothing on the desk changes.

import type { FoldModel } from '../../derive.ts';
import { useSay } from '../i18n/index.ts';
import { deskFold } from '../store/fold.ts';

export function FoldButton({ fold }: { fold: FoldModel }) {
  const say = useSay();
  return <button type="button" className="moretiles" data-qall="" aria-expanded={fold.open} onClick={() => deskFold.set(!fold.open)}>{say(fold.says)}</button>;
}
