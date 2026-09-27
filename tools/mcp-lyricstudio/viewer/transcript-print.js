'use strict';
document.getElementById('save-pdf').addEventListener('click', () => window.print());
// The export is already complete, independent of the viewer's paging/polling.
window.addEventListener('load', () => window.print(), { once: true });
