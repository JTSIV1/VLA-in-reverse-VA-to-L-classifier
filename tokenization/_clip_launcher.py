"""Launcher that mocks broken torchvision before importing transformers."""
import sys
import types
import importlib
import os

os.environ['USE_TF'] = '0'

# Mock torchvision to avoid torch 2.8 / torchvision 0.17 mismatch.
# We only use CLIP text encoder (no vision), so torchvision is not needed.
_tv = types.ModuleType('torchvision')
_tv.__spec__ = importlib.machinery.ModuleSpec('torchvision', None)
_tv.__version__ = '0.23.0'

# Create all submodules that transformers might reference
_submodules = [
    'transforms', 'transforms.functional', 'io', 'datasets',
    'models', 'ops', 'utils',
]
for sub in _submodules:
    full = f'torchvision.{sub}'
    m = types.ModuleType(full)
    m.__spec__ = importlib.machinery.ModuleSpec(full, None)
    sys.modules[full] = m
    # Set as attribute on parent
    parts = sub.split('.')
    parent = _tv
    for p in parts[:-1]:
        parent = sys.modules[f'torchvision.{p}']
    setattr(parent, parts[-1], m)

# InterpolationMode with all attributes transformers references
sys.modules['torchvision.transforms'].InterpolationMode = type(
    'InterpolationMode', (), {
        'BILINEAR': 2, 'BICUBIC': 3, 'NEAREST': 0, 'NEAREST_EXACT': 0,
        'LANCZOS': 1, 'BOX': 4, 'HAMMING': 5,
    })

sys.modules['torchvision'] = _tv

# Now import and run the actual module
from clip_action_language import parse_args, fit_clip_tokenizer

if __name__ == '__main__':
    args = parse_args()
    fit_clip_tokenizer(args)
