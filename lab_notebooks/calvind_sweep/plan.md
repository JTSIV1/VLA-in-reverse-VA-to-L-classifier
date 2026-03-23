# CALVIN-D Tokenizer Sweep

Goal: systematically compare action tokenizers on CALVIN D->D, measuring both
reconstruction quality and verb decodability, then use the best verb-optimized
tokenizers for downstream OpenVLA-mini finetuning.

## Stage 1: Train tokenizers (vanilla + aux loss sweep)

### Non-gradient tokenizers (one-off)

| Tokenizer | Type | Script |
|-----------|------|--------|
| raw       | native 7-DoF | -- (no tokenizer) |
| bin       | per-dim binning | -- (analytical) |
| FAST      | DCT + BPE | `train_tokenizer.py --tokenizer fast` |

### Gradient-based tokenizers (full sweep)

For each of VQ-BeT, OAT, QueST, train with all aux loss configs:

| Condition | verb_cls_lambda | clip_lambda |
|-----------|----------------|-------------|
| vanilla   | 0              | 0           |
| verb_0.01 | 0.01           | 0           |
| verb_0.1  | 0.1            | 0           |
| verb_0.5  | 0.5            | 0           |
| verb_1.0  | 1.0            | 0           |
| clip_0.1  | 0              | 0.1         |
| clip_0.5  | 0              | 0.5         |
| clip_1.0  | 0              | 1.0         |
| clip_2.0  | 0              | 2.0         |

Total gradient jobs: 3 tokenizers x 9 conditions = 27 jobs

## Stage 2: Verb decodability probe (motion-only)

For EVERY trained tokenizer from Stage 1 (including all lambda variants),
train the `ActionToVerbTransformer` in action_only mode with
`--weighted_loss --min_class_count 30` (21 classes).

This tells us:
- Which tokenizer type preserves verb info best (vanilla comparison)
- Whether aux losses improve verb decodability (lambda sweep)
- Whether CLIP loss helps as much as direct verb supervision

Probe targets: raw, bin, FAST, + 27 gradient tokenizer checkpoints = 30 total

## Stage 3: OpenVLA-mini finetuning

Take the best tokenizer from Stage 2 (by verb decodability improvement over
vanilla) and finetune OpenVLA-mini on CALVIN using that tokenizer vs. the
vanilla version.

## Code dependencies

- `tokenization/train_tokenizer.py` — unified tokenizer training (all stages)
- `datasets/calvin_dataset.py` — CalvinTokenizerDataset, CalvinActionCropDataset
- `verb_probe/train_transformer.py` — verb classification probe
- `tokenization/action_tokenizers.py` — TokenizerAdapter for loading trained tokenizers
