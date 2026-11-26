import { apiClient, ApiError, HttpStatus } from '@/services/httpClient';
import type { WorkletSynthesizer } from 'spessasynth_lib';
import { CUSTOM_SOUND_FONT, DEFAULT_SOUND_FONT, DEFAULT_SOUND_FONT_ID, readSoundFontPreference } from './soundFontPreference';

const DRUM_CHANNEL = 9;
const EXPRESSION = 11;

// Completed jobs use the GM family label. Live jobs use ace-midi's instrument name.
const PROGRAMS: ReadonlyMap<string, number> = new Map(Object.entries({
  piano: 0, acoustic_piano: 0, electric_piano: 2,
  'chromatic perc.': 8, chromatic_percussion: 8, organ: 16,
  guitar: 24, acoustic_guitar: 24, clean_electric_guitar: 26, distorted_electric_guitar: 29,
  bass: 32, acoustic_bass: 32, electric_bass: 33,
  strings: 40, violin: 40, viola: 41, cello: 42, contrabass: 43, orchestral_harp: 46, timpani: 47,
  ensemble: 48, string_ensemble: 48, synth_strings: 50, voice: 52, orchestra_hit: 55,
  brass: 56, trumpet: 56, trombone: 57, tuba: 58, french_horn: 60, brass_section: 61,
  reed: 64, soprano_and_alto_sax: 64, tenor_sax: 66, baritone_sax: 67, oboe: 68,
  english_horn: 69, bassoon: 70, clarinet: 71, pipe: 72, flutes: 72,
  'synth lead': 80, synth_lead: 80, 'synth pad': 88, synth_pad: 88,
  'synth fx': 96, ethnic: 104, percussive: 112, 'sound fx': 120,
}));

function programForFamily(family: string): number {
  const explicit = /^program[_ ](\d{1,3})$/.exec(family);
  const program = explicit ? Number(explicit[1]) : PROGRAMS.get(family) ?? 0;
  return program <= 127 ? program : 0;
}

export class SoundFontVoice {
  private channels = new Map<string, number>();
  private nextChannel = 0;
  private channelCount = 16;
  private timers = new Set<number>();

  private synth: WorkletSynthesizer;
  private context: AudioContext;

  private constructor(synth: WorkletSynthesizer, context: AudioContext) {
    this.synth = synth;
    this.context = context;
  }

  static async load(context: AudioContext, output: AudioNode, signal: AbortSignal): Promise<SoundFontVoice | null> {
    const preference = await readSoundFontPreference().catch(() => ({ selection: DEFAULT_SOUND_FONT, custom: undefined }));
    signal.throwIfAborted();
    const { selection, custom } = preference;
    const defaultBank = '/model-manager/midi-soundfont';
    const loadBank = async (url: string) => {
      try {
        const blob = await apiClient.get<Blob>(url, { signal, responseType: 'blob', timeout: 0 });
        return blob.arrayBuffer();
      } catch (error) {
        if (error instanceof ApiError && error.status === HttpStatus.notFound) return null;
        throw error;
      }
    };
    const useDefault = selection === DEFAULT_SOUND_FONT || selection === DEFAULT_SOUND_FONT_ID;
    const loadSelection = selection === CUSTOM_SOUND_FONT
      ? () => custom?.arrayBuffer()
      : () => loadBank(useDefault ? defaultBank : `${defaultBank}/${encodeURIComponent(selection)}`);
    const selectedBank = await loadSelection();
    const bank = selectedBank ?? (useDefault ? null : await loadBank(defaultBank));
    if (!bank) return null;
    signal.throwIfAborted();
    const [{ WorkletSynthesizer }, { default: processorUrl }] = await Promise.all([
      import('spessasynth_lib'),
      import('spessasynth_lib/dist/spessasynth_processor.min.js?url&no-inline'),
    ]);
    await context.audioWorklet.addModule(processorUrl);
    signal.throwIfAborted();
    const synth = new WorkletSynthesizer(context, { eventsEnabled: false });
    const destroy = () => synth.destroy();
    signal.addEventListener('abort', destroy, { once: true });
    try {
      await synth.isReady;
      try {
        await synth.soundBankManager.addSoundBank(bank, 'preview');
      } catch (error) {
        if (useDefault || !selectedBank || signal.aborted) throw error;
        const fallback = await loadBank(defaultBank);
        if (!fallback) throw error;
        await synth.soundBankManager.addSoundBank(fallback, 'preview');
      }
      signal.throwIfAborted();
      synth.setSystemParameter('effectsEnabled', false);
      synth.connect(output);
      return new SoundFontVoice(synth, context);
    } catch (error) {
      synth.destroy();
      throw error;
    } finally {
      signal.removeEventListener('abort', destroy);
    }
  }

  private channelFor(family: string, audible: boolean): number {
    const key = family.toLowerCase();
    const existing = this.channels.get(key);
    if (existing !== undefined) return existing;

    const channel = key === 'drums' ? DRUM_CHANNEL : this.nextChannel++;
    // SpessaSynth reserves channel 10 on every 16-channel MIDI port.
    if (this.nextChannel % 16 === DRUM_CHANNEL) this.nextChannel++;
    while (channel >= this.channelCount) {
      this.synth.addNewChannel();
      this.channelCount++;
    }
    this.channels.set(key, channel);
    this.synth.programChange(channel, programForFamily(key));
    this.setFamilyAudible(family, audible);
    return channel;
  }

  setFamilyAudible(family: string, audible: boolean): void {
    const channel = this.channels.get(family.toLowerCase());
    if (channel !== undefined) this.synth.controllerChange(channel, EXPRESSION, audible ? 127 : 0);
  }

  play(family: string, pitch: number, when: number, duration: number, audible: boolean): void {
    const channel = this.channelFor(family, audible);
    // Keep pending events cancellable when the transport pauses or seeks.
    this.schedule(when, () => this.synth.noteOn(channel, pitch, 80));
    this.schedule(when + duration, () => this.synth.noteOff(channel, pitch));
  }

  private schedule(when: number, action: () => void): void {
    const delay = (when - this.context.currentTime) * 1000;
    if (delay <= 0) {
      action();
      return;
    }
    const timer = window.setTimeout(() => {
      this.timers.delete(timer);
      action();
    }, delay);
    this.timers.add(timer);
  }

  stop(): void {
    for (const timer of this.timers) window.clearTimeout(timer);
    this.timers.clear();
    this.synth.stopAll(true);
  }

  dispose(): void {
    this.stop();
    this.synth.destroy();
  }
}
