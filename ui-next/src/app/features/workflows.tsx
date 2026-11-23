// Workflows, revisions, drafts and storage (package 7): what it draws in the shell, each in its own slot. The records
// and their steps are store/workflows.ts; the view is src/app/workflows/.

import { fill } from '../register/slots.ts';
import { WorkflowList } from '../workflows/List.tsx';
import { HeadButtons, SaveRow } from '../workflows/SaveRow.tsx';
import { DraftsButton } from '../workflows/Drafts.tsx';
import { SaveState, StoreToasts } from '../workflows/Storage.tsx';

fill('panel.workflows.head', HeadButtons);
fill('panel.workflows.save', SaveRow);
fill('panel.workflows', WorkflowList);
fill('centre.drafts', DraftsButton);
fill('centre.saveState', SaveState);
/* storage's toasts ride with the read-out, which is always drawn */
fill('centre.saveState', StoreToasts);
