"""
Compute verb-level Representational Similarity Matrices (RSMs) across language
models and vision-language models.

For each model, extracts an embedding for each of the 20 CALVIN verb classes,
then computes the 20x20 cosine similarity matrix (RSM).

Models:
  - Token embedding models (input embed layer only, CPU):
      gpt2, qwen2.5-0.5b
  - Full autoregressive models (last hidden state, GPU):
      vicuna-7b, llava-1.5-7b, qwen-vl-chat
  - Contrastive text encoders:
      clip-vit-b-32, siglip-so400m

Outputs:
  results/verb_rsms.npz   — {model_name: (20,20) cosine sim matrix}
  results/verb_embeds.npz — {model_name: (20, d) embedding matrix}
"""

import os
import sys
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# ── Verb list (20 sparse classes, excluding "left") ──────────────────────
VERBS = [
    "close", "grasp", "lift", "move", "open", "pick up", "place", "pull",
    "push", "put", "remove", "rotate", "slide", "stack", "store", "sweep",
    "take off", "turn", "turn off", "turn on",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HF_HOME = "/data/user_data/wenjiel2/.cache/huggingface"
os.environ.setdefault("HF_HOME", HF_HOME)
# Also check home cache
os.environ["HF_HUB_CACHE"] = os.environ.get(
    "HF_HUB_CACHE",
    ":".join([
        os.path.join(HF_HOME, "hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ])
)


def mean_pool_tokens(tokenizer, embed_matrix, verb):
    """Get embedding by mean-pooling input token embeddings."""
    ids = tokenizer.encode(verb, add_special_tokens=False)
    embs = embed_matrix[ids].float()
    return embs.mean(dim=0).numpy()


def last_hidden_state_embedding(model, tokenizer, verb, device="cuda"):
    """Get verb embedding as last hidden state (mean-pooled) from autoregressive model."""
    inputs = tokenizer(verb, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Use last hidden state, mean-pool over tokens (excluding padding)
    hidden = outputs.hidden_states[-1]  # (1, seq_len, d)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled[0].cpu().float().numpy()


def clip_text_embedding(model, processor, verb):
    """Get CLIP/SigLIP text embedding for a verb."""
    inputs = processor(text=[verb], return_tensors="pt", padding=True)
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs[0].cpu().float().numpy()


def siglip_text_embedding(model, processor, verb):
    """Get SigLIP text embedding for a verb."""
    inputs = processor(text=[verb], return_tensors="pt", padding="max_length", truncation=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs[0].cpu().float().numpy()


# ── Model loaders ────────────────────────────────────────────────────────

def load_gpt2():
    """GPT-2 token embeddings (CPU only)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    embed = model.transformer.wte.weight.detach()
    return "gpt2", lambda v: mean_pool_tokens(tokenizer, embed, v)


def load_qwen05b():
    """Qwen2.5-0.5B token embeddings (CPU only)."""
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    shard_path = hf_hub_download(model_id, "model.safetensors")
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        embed = f.get_tensor("model.embed_tokens.weight")
    return "qwen2.5-0.5b", lambda v: mean_pool_tokens(tokenizer, embed, v)


def load_vicuna():
    """Vicuna-7B-v1.5 last hidden state (GPU)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "lmsys/vicuna-7b-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return "vicuna-7b", lambda v: last_hidden_state_embedding(model, tokenizer, v)


def load_llava():
    """LLaVA-1.5-7B last hidden state from the LM backbone (GPU).

    The liuhaotian checkpoint contains full Llama/Vicuna weights merged in,
    so we load as LlamaForCausalLM to get the text-only LM backbone.
    """
    from transformers import AutoTokenizer, LlamaForCausalLM
    import warnings
    model_id = "liuhaotian/llava-v1.5-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress "llava -> llama" type warning
        model = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    model.eval()
    return "llava-1.5-7b", lambda v: last_hidden_state_embedding(model, tokenizer, v)


def load_qwen_vl():
    """Qwen-VL-Chat last hidden state from the LM backbone (GPU)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "Qwen/Qwen-VL-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return "qwen-vl-chat", lambda v: last_hidden_state_embedding(model, tokenizer, v)


def load_clip():
    """OpenAI CLIP ViT-B/32 text encoder."""
    from transformers import CLIPModel, CLIPProcessor
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval()
    return "clip-vit-b-32", lambda v: clip_text_embedding(model, processor, v)


def load_siglip():
    """Google SigLIP-SO400M text encoder."""
    from transformers import AutoModel, AutoProcessor
    model_id = "google/siglip-so400m-patch14-384"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    return "siglip-so400m", lambda v: siglip_text_embedding(model, processor, v)


# ── Main ─────────────────────────────────────────────────────────────────

MODEL_LOADERS = {
    "gpt2": load_gpt2,
    "qwen2.5-0.5b": load_qwen05b,
    "clip": load_clip,
    "siglip": load_siglip,
    "vicuna": load_vicuna,
    "llava": load_llava,
    "qwen-vl": load_qwen_vl,
}

# Order: small CPU models first, then GPU models (load one at a time)
MODEL_ORDER = ["gpt2", "qwen2.5-0.5b", "clip", "siglip", "vicuna", "llava", "qwen-vl"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER,
                        choices=list(MODEL_LOADERS.keys()),
                        help="Which models to compute RSMs for")
    args = parser.parse_args()

    all_rsms = {}
    all_embeds = {}

    for model_key in args.models:
        print("\n{'=' * 60}")
        print("Loading: {}".format(model_key))
        print("=" * 60)

        name, embed_fn = MODEL_LOADERS[model_key]()

        # Compute verb embeddings
        embeddings = []
        for v in VERBS:
            emb = embed_fn(v)
            embeddings.append(emb)
            print("  {:12s} -> shape {}".format(v, emb.shape))

        emb_matrix = np.stack(embeddings)  # (20, d)
        rsm = cosine_similarity(emb_matrix)  # (20, 20)

        all_embeds[name] = emb_matrix
        all_rsms[name] = rsm

        print("RSM shape: {}, embed shape: {}".format(rsm.shape, emb_matrix.shape))
        print("Mean off-diag sim: {:.4f}".format(
            rsm[np.triu_indices(len(VERBS), k=1)].mean()))

        # Free GPU memory for next model
        if model_key in ("vicuna", "llava", "qwen-vl"):
            import gc
            del embed_fn
            gc.collect()
            torch.cuda.empty_cache()
            print("  (freed GPU memory)")

    # Save (merge with existing results if any)
    rsm_path = os.path.join(RESULTS_DIR, "verb_rsms.npz")
    embed_path = os.path.join(RESULTS_DIR, "verb_embeds.npz")
    for path, new_data in [(rsm_path, all_rsms), (embed_path, all_embeds)]:
        if os.path.exists(path):
            existing = dict(np.load(path, allow_pickle=True))
            existing.update(new_data)
            existing["verbs"] = np.array(VERBS)
            np.savez(path, **existing)
        else:
            np.savez(path, **new_data, verbs=np.array(VERBS))
    print("\nSaved RSMs to {}".format(rsm_path))
    print("Saved embeddings to {}".format(embed_path))

    # Print RSM-to-RSM correlations (Spearman between upper triangles)
    # Load all RSMs (including previously computed ones)
    from scipy.stats import spearmanr
    all_rsms_full = dict(np.load(rsm_path, allow_pickle=True))
    all_rsms_full.pop("verbs", None)
    model_names = list(all_rsms_full.keys())
    all_rsms = all_rsms_full
    triu_idx = np.triu_indices(len(VERBS), k=1)

    print("\n── RSM-to-RSM Spearman correlations ──")
    header = "{:20s}".format("")
    for n in model_names:
        header += "{:>14s}".format(n)
    print(header)
    for n1 in model_names:
        row = "{:20s}".format(n1)
        for n2 in model_names:
            rho, _ = spearmanr(all_rsms[n1][triu_idx], all_rsms[n2][triu_idx])
            row += "{:14.3f}".format(rho)
        print(row)


if __name__ == "__main__":
    main()
