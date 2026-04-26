// spectralLifter.ts — Spectral Lifter subprocess wrapper
//
// Spawns the Python-based Spectral Lifter CLI to process a WAV file.
// Used as the first stage of the post-processing pipeline.

import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';

const execFileAsync = promisify(execFile);

// Path to the Spectral Lifter install
const SPECTRAL_LIFTER_DIR = path.resolve('D:\\Ace-Step-Latest\\Spectral-Lifter');
const CLI_SCRIPT = path.join(SPECTRAL_LIFTER_DIR, 'cli.py');

/**
 * Run Spectral Lifter on a WAV file.
 *
 * @param inputWav  - Path to input WAV file
 * @param outputWav - Path to write processed WAV file
 * @throws If the CLI script is missing, the input file doesn't exist, or processing fails
 */
export async function runSpectralLifter(inputWav: string, outputWav: string): Promise<void> {
  if (!fs.existsSync(CLI_SCRIPT)) {
    throw new Error(`Spectral Lifter CLI not found at ${CLI_SCRIPT}`);
  }
  if (!fs.existsSync(inputWav)) {
    throw new Error(`Input file not found: ${inputWav}`);
  }

  console.log(`[Spectral Lifter] Processing: ${path.basename(inputWav)} → ${path.basename(outputWav)}`);
  const start = Date.now();

  const { stderr } = await execFileAsync('python', [CLI_SCRIPT, inputWav, outputWav], {
    timeout: 300_000,  // 5 min max
    cwd: SPECTRAL_LIFTER_DIR,
  });

  if (stderr) {
    for (const line of stderr.split('\n')) {
      if (line.trim()) console.log(`[Spectral Lifter] ${line.trim()}`);
    }
  }

  if (!fs.existsSync(outputWav)) {
    throw new Error('Spectral Lifter produced no output file');
  }

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`[Spectral Lifter] Complete in ${elapsed}s`);
}
