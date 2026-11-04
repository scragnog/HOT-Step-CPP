"""Capture unmodified upstream CPU inference with observation-only hooks."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time
from common import Dumps, digest, load_upstream, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch = load_upstream(args.upstream)
    import librosa
    import numpy as np
    import soundfile as sf
    from mulacover._symbolic_transcription.melody import MelodyTranscriber
    from mulacover._symbolic_transcription.harmony import ChordTranscriber
    from mulacover.symbolic import SymbolicCondition

    info = sf.info(args.audio)
    if not 0 < info.duration < 60:
        raise ValueError('Each input must be nonempty and shorter than 60 seconds')
    dumps = Dumps(args.out)
    manifest = dict(status='running', input_sha256=digest(args.audio),
                    duration_seconds=info.duration, sample_rate=info.samplerate,
                    device='cpu', dtype='float32', threads=4, seed=0,
                    packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                    timings_seconds={}, skips=[])
    write_json(args.out / 'manifest.json', manifest)
    started = time.perf_counter()
    melody = MelodyTranscriber(args.weights / 'yourmt3/last.ckpt', 'cpu')
    model = melody.model
    write_json(args.out / 'yourmt3-config.json', dict(audio=model.audio_cfg, model=model.model_cfg,
               max_total_token_length=model.max_total_token_length))
    codec = model.task_manager.tokenizer.codec
    vocabulary = [asdict(codec.decode_event_index(i)) for i in range(model.task_manager.num_tokens)]
    write_json(args.out / 'event-vocabulary.json', vocabulary)
    write_json(args.out / 'program-channel-map.json',
               {str(k): {str(a): int(b) if isinstance(b, np.integer) else b for a, b in v.items()}
                for k, v in model.task_manager.program2channel_vocab.items()})
    state = dict(batch=-1, step=0)
    handles = []

    def spectrogram_input(module, inputs):
        state['batch'] += 1
        state['step'] = 0
        dumps.put(f"mt3.b{state['batch']}.waveform-padded", inputs[0])

    def activation(name, select=lambda output: output):
        def hook(module, inputs, output):
            dumps.put(f"mt3.b{state['batch']}.{name}", select(output))
        return hook

    def logits(module, inputs, output):
        step = state['step']
        if step < 2:
            dumps.put(f"mt3.b{state['batch']}.{'prefill' if step == 0 else 'cached-step'}.logits", output)
        state['step'] += 1

    handles.append(model.spectrogram.register_forward_pre_hook(spectrogram_input))
    handles.append(model.spectrogram.register_forward_hook(activation('linear-spectrogram')))
    handles.append(model.pre_encoder.register_forward_hook(activation('pre-encoder')))
    handles.append(model.encoder.register_forward_hook(activation('encoder', lambda x: x['last_hidden_state'])))
    handles.append(model.pre_decoder.register_forward_hook(activation('pre-decoder')))
    handles.append(model.lm_head.register_forward_hook(logits))
    original_inference = model.inference
    token_batches = []

    def inference(audio):
        result = original_inference(audio)
        token_batches.append(result.cpu().tolist())
        return result

    model.inference = inference
    notes = melody.transcribe(args.audio)
    for handle in handles:
        handle.remove()
    manifest['timings_seconds']['yourmt3_load_and_inference'] = time.perf_counter() - started
    serialized_notes = [asdict(n) for n in notes]
    write_json(args.out / 'tokens-by-batch-channel.json', token_batches)
    write_json(args.out / 'merged-notes-seconds.json', serialized_notes)
    write_json(args.out / 'vocal-program-100-seconds.json', [n for n in serialized_notes if not n['is_drum'] and n['program'] == 100])
    write_json(args.out / 'drums-seconds.json', [n for n in serialized_notes if n['is_drum']])
    print('YourMT3 complete', manifest['timings_seconds'], flush=True)
    write_json(args.out / 'manifest.json', manifest)
    del melody, model

    started = time.perf_counter()
    audio, sr = librosa.load(str(args.audio), sr=None)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    bpm = float(np.asarray(tempo).item())
    manifest['librosa_bpm'] = bpm
    chords = ChordTranscriber(args.weights / 'chord', 'cpu')
    original_load, original_cqt = librosa.load, librosa.hybrid_cqt
    original_tuning = librosa.estimate_tuning
    # hybrid_cqt resolves estimate_tuning in its defining module.
    import librosa.core.constantq as constantq
    original_internal_tuning = constantq.estimate_tuning

    def load(*a, **kw):
        result = original_load(*a, **kw)
        dumps.put('chord.waveform-22050', result[0])
        return result

    def tuning(*a, **kw):
        result = original_internal_tuning(*a, **kw)
        manifest['tuning_bins'] = float(result)
        return result

    def cqt(*a, **kw):
        result = original_cqt(*a, **kw)
        dumps.put('chord.hybrid-cqt', result.T)
        dumps.put('chord.cqt-magnitude', np.abs(result.T).astype(np.float32))
        return result

    fold_probabilities = []
    for fold, model in enumerate(chords.models):
        def cnn_input(module, inputs, fold=fold):
            dumps.put(f'chord.fold{fold}.crop-18-270', inputs[0])
        def cnn_output(module, inputs, output, fold=fold):
            dumps.put(f'chord.fold{fold}.cnn', output)
        def lstm_output(module, inputs, output, fold=fold):
            dumps.put(f'chord.fold{fold}.lstm', output[0])
        model.audio_feature_block.register_forward_pre_hook(cnn_input)
        model.audio_feature_block.register_forward_hook(cnn_output)
        model.lstm1.register_forward_hook(lstm_output)
        original = model.inference
        def predict(features, original=original, fold=fold):
            result = original(features)
            fold_probabilities.append(result)
            for head, values in enumerate(result):
                dumps.put(f'chord.fold{fold}.head{head}', values)
            return result
        model.inference = predict
    original_decode = chords.decoder.decode_segments
    seconds_segments = []

    def decode(probabilities, frame_seconds):
        for head, values in enumerate(probabilities):
            dumps.put(f'chord.mean.head{head}', values)
        result = original_decode(probabilities, frame_seconds)
        seconds_segments.extend(result)
        write_json(args.out / 'chords-seconds.json', result)
        write_json(args.out / 'chord-frame-labels.json', chords.decoder.decode(probabilities))
        return result

    chords.decoder.decode_segments = decode
    librosa.load, librosa.hybrid_cqt = load, cqt
    constantq.estimate_tuning = tuning
    # No tempo fallback is silently substituted for a silent input.
    try:
        if bpm <= 0 or not np.isfinite(bpm):
            manifest['skips'].append('No positive librosa BPM; upstream rejects transcription')
            chord_events = None
        else:
            chord_events = chords.transcribe(args.audio, bpm)
    finally:
        librosa.load, librosa.hybrid_cqt = original_load, original_cqt
        librosa.estimate_tuning = original_tuning
        constantq.estimate_tuning = original_internal_tuning
    manifest['timings_seconds']['chord_load_features_inference_and_bpm'] = time.perf_counter() - started
    if chord_events is not None:
        write_json(args.out / 'chords-rounded-beats.json', chord_events)
        try:
            condition = SymbolicCondition.from_transcription(notes, chord_events, bpm)
        except ValueError as error:
            if str(error) != 'Melody must contain at least one note.':
                raise
            manifest['skips'].append(str(error))
        else:
            write_json(args.out / 'triplets-sixteenths.json',
                       {k: getattr(condition, k).tolist() for k in ('melody', 'chords', 'drums')})
            for name, tensor in condition.to_tensors().items():
                dumps.put('roll.' + name, tensor)
            condition.save_midi(args.out / 'midi')
    # The seconds variant is an event representation. Dense rolls require a
    # time-grid policy; do not invent one and label it upstream.
    write_json(args.out / 'seconds-variant.json', dict(notes=serialized_notes, chords=seconds_segments))
    manifest['status'] = 'captured' if not manifest['skips'] else 'captured-with-upstream-rejection'
    manifest['tensor_count'] = len(dumps.entries)
    write_json(args.out / 'manifest.json', manifest)
    print(json.dumps(manifest['timings_seconds']), manifest['status'], flush=True)


if __name__ == '__main__':
    main()
