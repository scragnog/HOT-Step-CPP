"""Load all six checkpoints strictly on CPU and record model parameter counts."""
import argparse
import importlib.metadata
from pathlib import Path
from common import load_upstream, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch = load_upstream(args.upstream)
    from mulacover._symbolic_transcription.melody import MelodyTranscriber
    from mulacover._symbolic_transcription.harmony import ChordTranscriber
    result = {'device': 'cpu', 'dtype': 'float32', 'threads': 4, 'models': {}}
    loaded_states = []
    original_load = torch.load
    def observe_load(*a, **kw):
        checkpoint = original_load(*a, **kw)
        state = checkpoint.get('state_dict', checkpoint.get('net', checkpoint))
        loaded_states.append({'file': Path(a[0]).name, 'tensors': [
            {'name': n, 'shape': list(t.shape), 'dtype': str(t.dtype)}
            for n, t in state.items() if isinstance(t, torch.Tensor)]})
        return checkpoint
    torch.load = observe_load
    for name, model in [('yourmt3', MelodyTranscriber(args.weights / 'yourmt3/last.ckpt', 'cpu').model)]:
        result['models'][name] = describe(model)
        write_json(args.out / 'inventory.json', result)
    chords = ChordTranscriber(args.weights / 'chord', 'cpu')
    for fold, model in enumerate(chords.models):
        result['models'][f'chord-{fold}'] = describe(model)
    result['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    result['serialized_state_dicts'] = loaded_states
    torch.load = original_load
    write_json(args.out / 'inventory.json', result)
    print({k: v['parameters'] for k, v in result['models'].items()}, flush=True)


def describe(model):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'state_elements': sum(p.numel() for p in model.state_dict().values()),
            'tensors': [{'name': n, 'shape': list(t.shape), 'dtype': str(t.dtype)}
                        for n, t in model.state_dict().items()]}


if __name__ == '__main__':
    main()
