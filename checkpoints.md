# Calvin VLA Checkpoint Locations

The finetuned Vision-Language-Action (VLA) models for the Calvin dataset are primarily stored in the following locations within the project data area:

### MiniVLA (from-scratch Qwen2.5-0.5B)
- **Base Path**: `/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/runs/calvind_scratch/`
- **Structure**: Each training condition (e.g., `bin_baseline`, `vb_c5e16g4`, `quest_h16f256d2`) has a dedicated subfolder.
- **Weights**: Checkpoints are `.pt` files located in the `checkpoints/` subdirectory of each run folder.

### OpenVLA (7B + LoRA)
- **Base Path**: `/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/runs/calvind_policy/`
- **Weights**: These folders contain LoRA adapter weights (`adapter_model.bin`) and processor configurations.

### Tokenizers (VQ-Bet / QueST / OAT)
- **Base Path**: `/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/calvind_hp_sweep/`
- **Description**: Contains the `full.pth` and `tokenizer_weights.pth` files for various tokenizer configurations used during the sweeps.

---
*Note: The model evaluation logic in `policy/eval_policy.py` automatically resolves these paths based on the model family and condition provided.*
