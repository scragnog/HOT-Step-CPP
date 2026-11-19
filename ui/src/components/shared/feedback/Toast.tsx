// Toast.tsx — the app's one way to show a toast.
//
// Everything goes to the react-toastify stack mounted in main.tsx: top right,
// stacked, auto-dismissing, paused on hover. Studios used to draw their own
// toast boxes in assorted corners; call notify() instead.

import { toast } from 'react-toastify';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

// ponytail: a word match decides the colour for callers that pass no type,
// so the studios' old string-only showToast calls keep working; pass a type
// where the wording is ambiguous.
const FAILURE = /\b(fail(ed|ure)?|error|not found|invalid|lost|not authenticated|could not|can't|cannot)\b/i;

/** Show a toast. Without a type, a message that reads as a failure is shown as an error. */
export function notify(message: string, type?: ToastType): void {
  toast(message, { type: type ?? (FAILURE.test(message) ? 'error' : 'default') });
}
