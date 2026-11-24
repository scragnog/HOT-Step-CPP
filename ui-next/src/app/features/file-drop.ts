// A file dragged over the page can be dropped only on a drop zone (13-files.js): anywhere else the page takes it and
// says where to drop it, rather than the browser opening the file in place of the page. A drop zone takes its own
// files (parts/DropZone.tsx).

import { dropOutside } from '../../derive.ts';
import { useDesk } from '../store/desk.ts';

const files = (e: DragEvent) => [...(e.dataTransfer?.types ?? [])].includes('Files');
const onZone = (e: DragEvent) => !!(e.target as Element | null)?.closest?.('label.drop');

document.addEventListener('dragover', e => { if (!files(e) || onZone(e)) return; e.preventDefault(); if (e.dataTransfer) e.dataTransfer.dropEffect = 'none'; });
document.addEventListener('drop', e => { if (!files(e) || onZone(e)) return; e.preventDefault(); useDesk.getState().showToast({ text: dropOutside() }); });
