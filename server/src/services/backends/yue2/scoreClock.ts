import abcjs from 'abcjs';
import { barsIn, isVocalVoice, scoreBarSegments } from './scoreHealth.js';

export interface ScoreClockBar {
  start: number;
  end: number;
  wholeStart: number;
  wholeEnd: number;
  meter: string;
  bpm: number;
  beatWholeNotes: number;
}

/** Walk the active SheetSage M:/Q: fields over Vocal bars. */
export function scoreBarClock(abc: string): ScoreClockBar[] {
  const bars: ScoreClockBar[] = [];
  let meter = '';
  let meterWholeNotes = 0;
  let bpm = 0;
  let beatWholeNotes = 0;
  let inVocal = false;
  let seconds = 0;
  let wholeNotes = 0;
  for (const raw of abc.split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('%')) continue;
    if (line.startsWith('M:')) {
      const match = /^M:\s*(\d+)\s*\/\s*(\d+)\s*$/.exec(line);
      if (!match || Number(match[2]) === 0 || Number(match[1]) === 0) throw new Error('Unsupported score M: meter.');
      meter = `${Number(match[1])}/${Number(match[2])}`;
      meterWholeNotes = Number(match[1]) / Number(match[2]);
      continue;
    }
    if (line.startsWith('Q:')) {
      const match = /^Q:\s*(?:(\d+)\s*\/\s*(\d+)\s*=\s*)?(\d+(?:\.\d+)?)\s*$/.exec(line);
      if (!match || (match[2] && Number(match[2]) === 0) || Number(match[3]) <= 0) {
        throw new Error('Unsupported score Q: tempo.');
      }
      beatWholeNotes = match[1] ? Number(match[1]) / Number(match[2]) : 1 / 4;
      bpm = Number(match[3]);
      continue;
    }
    if (line.startsWith('V:')) { inVocal = isVocalVoice(line); continue; }
    if (/^[A-Za-z]:/.test(line) || !inVocal) continue;
    if (!meterWholeNotes || !bpm || !beatWholeNotes) throw new Error('Vocal score lacks an active M: meter or Q: tempo.');
    for (const segment of scoreBarSegments(line)) {
      for (let i = 0; i < barsIn(segment); i++) {
        const duration = meterWholeNotes / beatWholeNotes * 60 / bpm;
        bars.push({ start: seconds, end: seconds + duration, wholeStart: wholeNotes,
          wholeEnd: wholeNotes + meterWholeNotes, meter, bpm, beatWholeNotes });
        seconds += duration;
        wholeNotes += meterWholeNotes;
      }
    }
  }
  if (!bars.length) throw new Error('Score has no Vocal bars.');
  return bars;
}

export function scoreSecondsAtWhole(bars: ScoreClockBar[], position: number): number {
  const bar = bars.find(item => position >= item.wholeStart - 1e-7 && position < item.wholeEnd - 1e-7) ?? bars.at(-1)!;
  return bar.start + (position - bar.wholeStart) / bar.beatWholeNotes * 60 / bar.bpm;
}

/** First sounding Vocal event in each score bar, using ABC note/rest durations. */
export function scoreVocalNoteOnsets(abc: string, bars: ScoreClockBar[]): Array<number | null> {
  const tune = abcjs.parseOnly(abc)[0];
  if (!tune) throw new Error('ABC parser returned no tune.');
  const onsets: Array<number | null> = Array(bars.length).fill(null);
  let barIndex = 0;
  let offset = 0;
  let spanBars = 1;
  for (const line of tune.lines) {
    // SheetSage puts Vocal on the first staff; Ins is a separate staff.
    for (const event of line.staff?.[0]?.voices?.[0] ?? []) {
      if (event.el_type === 'note') {
        const duration = Number(event.duration);
        if (!(duration > 0) || !Number.isFinite(duration)) throw new Error('Invalid Vocal note duration.');
        if (barIndex >= bars.length) throw new Error(`ABC notes extend beyond the Vocal bar grid at bar ${barIndex + 1}, char ${event.startChar}.`);
        if (event.pitches?.length && onsets[barIndex] === null) {
          onsets[barIndex] = scoreSecondsAtWhole(bars, bars[barIndex].wholeStart + offset);
        }
        if (event.rest?.type === 'multimeasure') {
          spanBars = Number(event.rest.text);
          if (!Number.isInteger(spanBars) || spanBars < 1) throw new Error('Invalid Vocal multibar rest.');
        }
        offset += duration;
      } else if (event.el_type === 'bar' && barIndex < bars.length) {
        // ABC's Z4 duration is not consistently meter-scaled; use its bar count.
        barIndex += spanBars;
        offset = 0;
        spanBars = 1;
      }
    }
  }
  if (barIndex !== bars.length) throw new Error('ABC notes and Vocal bar grid disagree.');
  return onsets;
}

/** Return a fractional bar index for an audio time, extrapolating at the edges. */
export function scoreBarPosition(bars: ScoreClockBar[], seconds: number): number {
  const index = bars.findIndex(bar => seconds < bar.end);
  if (index >= 0) return index + (seconds - bars[index].start) / (bars[index].end - bars[index].start);
  const last = bars.at(-1)!;
  return bars.length + (seconds - last.end) / (last.end - last.start);
}
