// contracts/shutdown.ts — wire shapes for /api/shutdown (server/src/routes/shutdown.ts).

export interface ShutdownResponse { success: boolean; message: string }
export type RestartResponse = ShutdownResponse;
