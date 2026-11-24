// The VST3 chain's "+ Add" (parts.ts ACTIONS vstHost): the VST3 host is another page of the real application, which
// this page does not draw; until it is wired in (package 10), the button says so, as the concept's does.

import { useDesk } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';
import { defineAction } from '../register/actions.ts';

defineAction('vstHost', () => useDesk.getState().showToast({ text: ui('action.vstHost') }));
