// logger.ts — File-based logging system
//
// Mirrors the hot-step-9000 logging architecture:
//   logs/<session-timestamp>/
//     ├── node_console.log       — all console output (mirrored transparently)
//     ├── ace_engine.log         — ace-server stdout/stderr
//     └── generations/
//         └── gen_<jobId>_<type>.log  — per-generation logs

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../../..');

/** Current session log directory (null if not initialized) */
let sessionDir: string | null = null;
let generationsDir: string | null = null;
let consoleLogStream: fs.WriteStream | null = null;
let engineLogStream: fs.WriteStream | null = null;

/** Per-generation log buffers: jobId → lines[] */
const generationBuffers = new Map<string, string[]>();

function timestamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
}

function isoTimestamp(): string {
  return new Date().toISOString();
}

/**
 * Initialize the logging system. Call once at server startup, before any
 * console output you care about.
 */
export function initLogger(): string {
  const logsRoot = path.join(projectRoot, 'logs');
  if (!fs.existsSync(logsRoot)) {
    fs.mkdirSync(logsRoot, { recursive: true });
  }

  sessionDir = path.join(logsRoot, timestamp());
  fs.mkdirSync(sessionDir, { recursive: true });

  generationsDir = path.join(sessionDir, 'generations');
  fs.mkdirSync(generationsDir, { recursive: true });

  // Open console log stream
  const consoleLogPath = path.join(sessionDir, 'node_console.log');
  consoleLogStream = fs.createWriteStream(consoleLogPath, { flags: 'a' });

  // Open engine log stream
  const engineLogPath = path.join(sessionDir, 'ace_engine.log');
  engineLogStream = fs.createWriteStream(engineLogPath, { flags: 'a' });

  // Hook stdout/stderr to mirror transparently (same pattern as hot-step-9000)
  const originalStdoutWrite = process.stdout.write.bind(process.stdout);
  const originalStderrWrite = process.stderr.write.bind(process.stderr);

  (process.stdout as any).write = (chunk: any, encoding?: any, callback?: any) => {
    try { consoleLogStream?.write(chunk); } catch { /* ignore */ }
    return originalStdoutWrite(chunk, encoding, callback);
  };

  (process.stderr as any).write = (chunk: any, encoding?: any, callback?: any) => {
    try { consoleLogStream?.write(chunk); } catch { /* ignore */ }
    return originalStderrWrite(chunk, encoding, callback);
  };

  consoleLogStream.write(`[Logger] Session started at ${isoTimestamp()}\n`);
  consoleLogStream.write(`[Logger] Log directory: ${sessionDir}\n`);

  return sessionDir;
}

/**
 * Write a line to the ace-engine log file.
 * Call this from the ace-server child process stdout/stderr handlers.
 */
export function logEngine(line: string): void {
  if (!engineLogStream) return;
  try {
    engineLogStream.write(line.endsWith('\n') ? line : line + '\n');
  } catch { /* ignore */ }
}

/**
 * Start capturing log lines for a specific generation job.
 * Returns the job's log file path.
 */
export function startGenerationLog(jobId: string, taskType: string = 'text2music'): string | null {
  if (!generationsDir) return null;

  generationBuffers.set(jobId, []);

  const header = [
    `${isoTimestamp()} | INFO    | ============================================================`,
    `${isoTimestamp()} | INFO    | GENERATION STARTED: Job ${jobId}`,
    `${isoTimestamp()} | INFO    | Task Type: ${taskType}`,
  ];
  generationBuffers.get(jobId)!.push(...header);

  return path.join(generationsDir, `gen_${jobId}_${taskType}.log`);
}

/**
 * Append a line to a generation's log buffer.
 */
export function logGeneration(jobId: string, level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', message: string): void {
  const buf = generationBuffers.get(jobId);
  if (!buf) return;
  buf.push(`${isoTimestamp()} | ${level.padEnd(7)} | ${message}`);
}

/**
 * Log a full params object (JSON pretty-printed) to a generation log.
 */
export function logGenerationParams(jobId: string, params: Record<string, any>): void {
  const buf = generationBuffers.get(jobId);
  if (!buf) return;
  buf.push(`${isoTimestamp()} | INFO    | Parameters:`);
  const json = JSON.stringify(params, null, 2);
  for (const line of json.split('\n')) {
    buf.push(`${isoTimestamp()} | INFO    | ${line}`);
  }
  buf.push(`${isoTimestamp()} | INFO    | ============================================================`);
}

/**
 * Finalize and flush a generation log to disk.
 * Call when a generation completes or fails.
 */
export function finishGenerationLog(jobId: string, taskType: string = 'text2music'): void {
  if (!generationsDir) return;

  const buf = generationBuffers.get(jobId);
  if (!buf) return;

  buf.push(`${isoTimestamp()} | INFO    | GENERATION COMPLETED.`);

  const logPath = path.join(generationsDir, `gen_${jobId}_${taskType}.log`);
  try {
    fs.writeFileSync(logPath, buf.join('\n') + '\n');
    console.log(`[Logger] Generation log saved: ${path.relative(projectRoot, logPath)}`);
  } catch (e) {
    console.error(`[Logger] Failed to write generation log: ${e}`);
  }

  generationBuffers.delete(jobId);
}

/**
 * Finalize a generation log as failed.
 */
export function failGenerationLog(jobId: string, error: string, taskType: string = 'text2music'): void {
  if (!generationsDir) return;

  const buf = generationBuffers.get(jobId);
  if (!buf) return;

  buf.push(`${isoTimestamp()} | ERROR   | GENERATION FAILED: ${error}`);

  const logPath = path.join(generationsDir, `gen_${jobId}_${taskType}.log`);
  try {
    fs.writeFileSync(logPath, buf.join('\n') + '\n');
    console.log(`[Logger] Generation log (failed) saved: ${path.relative(projectRoot, logPath)}`);
  } catch (e) {
    console.error(`[Logger] Failed to write generation log: ${e}`);
  }

  generationBuffers.delete(jobId);
}

/**
 * Get the current session log directory, or null if not initialized.
 */
export function getSessionDir(): string | null {
  return sessionDir;
}

/**
 * Close all log streams. Call during shutdown.
 */
export function closeLogger(): void {
  try { consoleLogStream?.end(); } catch { /* ignore */ }
  try { engineLogStream?.end(); } catch { /* ignore */ }
  consoleLogStream = null;
  engineLogStream = null;
}
