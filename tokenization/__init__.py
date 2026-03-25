"""Action tokenizers for CALVIN / Bridge trajectories.

Modules:
    train_tokenizer  - Unified tokenizer training script
    eval_tokenizer   - Evaluation (recon, verb, CLIP, codebook util)
    aux_heads        - Auxiliary heads (VerbHead, ContrastiveHead, TextEncoder)

Vendored tokenizer codebases:
    oat/             - OAT/QueST tokenizer (register encoder + FSQ)
    vq_bet_official/ - VQ-BeT tokenizer (MLP + ResidualVQ)
    fast/            - FAST tokenizer (DCT + BPE)
"""
