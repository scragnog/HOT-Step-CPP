/**
 * The human message for a thrown value.
 *
 * Nearly every catch block in the UI wrote `err instanceof Error ? err.message
 * : String(err)` by hand. A thrown string, a thrown object and an `Error` with
 * an empty message all need the same answer in one place, so a failure message
 * reads the same in a toast, a card and the training store.
 */
export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message || String(err);
  return String(err);
}
