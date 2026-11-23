// A node's actions as links or buttons. What an action does is the core's (runAction): a change, which is committed
// as every step is; going to a tier and ringing controls; adding a module; or a named action the shell runs (runNamed
// below, the handlers for parts.ts ACTIONS). Their words are the core's (actionWords, stripWords).

import { actionWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import { doAction } from './doAction.ts';
import type { ViewProps } from './types.ts';

export function ActionLinks({ def, node, nodeKey, look }: ViewProps & { look: 'link' | 'button' }) {
  const say = useSay(), desk = useDesk(s => s.desk);
  return (
    <>
      {actionWords(ix, node, desk, nodeKey).map(({ action: a, label, tip }, i) => {
        const ring = a.do === 'tier' && a.ring ? a.ring.join(' ') : undefined;
        return look === 'link'
          ? <button key={i} type="button" className="linkbtn" data-tier={a.do === 'tier' ? a.tier : undefined} data-ring={ring} title={tip ? say(tip) : undefined} onClick={() => doAction(def, a, node)}>{say(label)}</button>
          : <button key={i} type="button" className="pbtn" title={tip ? say(tip) : undefined} onClick={() => doAction(def, a, node)}>{say(label)}</button>;
      })}
    </>
  );
}
