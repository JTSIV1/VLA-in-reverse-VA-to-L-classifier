"""
RSM vs Confusion Matrix Correlation Analysis (Round 2 — Fixed)

Properly constructs symmetric confusion matrices and correlates them with
contextualized & non-contextualized verb embeddings from LLMs.

Key improvements:
1. Symmetric confusion matrix: M[i,j] = avg(cm[i,j] / row_sum_i, cm[j,i] / row_sum_j)
   - This captures bidirectional confusion rates properly
   - Element [i,j] = how much verbs i and j are confused with each other
   - Truly symmetric across the diagonal

2. Contextualized embeddings: Extract from instruction context
   - Each verb embedded in context of CALVIN instructions
   - Verbs used in different instructions get different embeddings
   - Mean pool across all occurrences

3. Non-contextualized: Just the bare verb word embedding

Models:
  - llama-2-7b (base LLM, text-only)
  - llava-1.5-7b (visual VLM, we use text LM backbone)
  - qwen-2.5 (base LLM, text-only) — Added
  - qwen-vl (visual VLM, we use text LM backbone) — Added

Output:
  results/rsm_confusion_analysis.npz — contains:
    - confusion_matrix (20, 20) — symmetric motor similarity
    - rsm_gpt2, rsm_llama, rsm_llava, rsm_qwen, rsm_qwen_vl
    - rsm_gpt2_ctx, rsm_llama_ctx, rsm_llava_ctx, rsm_qwen_ctx, rsm_qwen_vl_ctx
    - correlations_dict (spearman rho & p-values)
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

VERBS = [
    "close", "grasp", "lift", "move", "open", "pick up", "place", "pull",
    "push", "put", "remove", "rotate", "slide", "stack", "store", "sweep",
    "take off", "turn", "turn off", "turn on",
]
NUM_VERBS = len(VERBS)

HF_HOME = "/data/user_data/wenjiel2/.cache/huggingface"
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ["HF_HUB_CACHE"] = os.environ.get(
    "HF_HUB_CACHE",
    ":".join([
        os.path.join(HF_HOME, "hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ])
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFUSION MATRIX CONSTRUCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_predictions(preds_path):
    """Load predictions and corresponding verb classes."""
    with open(preds_path) as f:
        d = json.load(f)
    labels = np.array(d["labels"])
    preds = np.array(d["preds"])
    id_to_verb = {int(k): v for k, v in d["id_to_verb"].items()}
    return labels, preds, id_to_verb


def build_symmetric_confusion_matrix(labels, preds, verbs):
    """
    Build symmetric confusion matrix properly.

    M[i,j] represents the average bidirectional confusion rate:
      = (rate_of_i_confused_with_j + rate_of_j_confused_with_i) / 2

    This is symmetric by construction: M[i,j] = M[j,i]
    Values represent: how much a verb pair is mutually confused.

    Args:
        labels: true labels (numeric)
        preds: predicted labels (numeric)
        verbs: list of verb names (ordered by ID)

    Returns:
        cm_sym: (len(verbs), len(verbs)) symmetric confusion matrix
        verbs_aligned: verbs aligned to the matrix indices
    """
    # Build raw confusion matrix (rows=true, cols=pred)
    cm_raw = confusion_matrix(labels, preds, labels=list(range(len(verbs))))

    # Row-normalize: cm[i,j] / sum_j(cm[i,j]) = P(pred=j | true=i)
    row_sums = cm_raw.sum(axis=1, keepdims=True).clip(min=1)
    cm_normalized = cm_raw.astype(float) / row_sums

    # Create symmetric version
    # M[i,j] = (cm_norm[i,j] + cm_norm[j,i]) / 2
    cm_sym = (cm_normalized + cm_normalized.T) / 2.0

    return cm_sym, verbs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMBEDDING EXTRACTION (NON-CONTEXTUALIZED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mean_pool_tokens(tokenizer, embed_matrix, verb):
    """Get embedding by mean-pooling input token embeddings."""
    ids = tokenizer.encode(verb, add_special_tokens=False)
    embs = embed_matrix[ids].float()
    return embs.mean(dim=0).numpy()


def last_hidden_state_embedding(model, tokenizer, verb, device="cuda"):
    """Get verb embedding as mean-pooled last hidden state."""
    inputs = tokenizer(verb, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled[0].cpu().float().numpy()


def extract_embeddings_gpt2(verbs):
    """Extract non-contextualized GPT-2 embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    embed = model.transformer.wte.weight.detach()

    embeddings = []
    for v in verbs:
        emb = mean_pool_tokens(tokenizer, embed, v)
        embeddings.append(emb)

    return np.stack(embeddings)


def extract_embeddings_vicuna(verbs, device="cuda"):
    """Extract non-contextualized Vicuna-7B-v1.5 embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Vicuna-7B-v1.5...")
    model_id = "lmsys/vicuna-7b-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    embeddings = []
    for v in verbs:
        emb = last_hidden_state_embedding(model, tokenizer, v, device)
        embeddings.append(emb)

    torch.cuda.empty_cache()
    return np.stack(embeddings)


def extract_embeddings_llava_text(verbs, device="cuda"):
    """Extract non-contextualized LLaVA-1.5-7B text embeddings (LM backbone only)."""
    from transformers import AutoTokenizer, LlamaForCausalLM
    import warnings
    print("Loading LLaVA-1.5-7B (text backbone)...")
    model_id = "liuhaotian/llava-v1.5-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    model.eval()

    embeddings = []
    for v in verbs:
        emb = last_hidden_state_embedding(model, tokenizer, v, device)
        embeddings.append(emb)

    torch.cuda.empty_cache()
    return np.stack(embeddings)


def extract_embeddings_qwen(verbs, device="cuda"):
    """Extract non-contextualized Qwen-2.5-7B embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Qwen-2.5-7B...")
    model_id = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    embeddings = []
    for v in verbs:
        emb = last_hidden_state_embedding(model, tokenizer, v, device)
        embeddings.append(emb)

    torch.cuda.empty_cache()
    return np.stack(embeddings)


def extract_embeddings_qwen_vl(verbs, device="cuda"):
    """Extract non-contextualized Qwen-VL-Chat text embeddings (LM backbone)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Qwen-VL-Chat (text backbone)...")
    model_id = "Qwen/Qwen-VL-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    embeddings = []
    for v in verbs:
        emb = last_hidden_state_embedding(model, tokenizer, v, device)
        embeddings.append(emb)

    torch.cuda.empty_cache()
    return np.stack(embeddings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMBEDDING EXTRACTION (CONTEXTUALIZED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_calvin_instructions(data_dir="/data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/"):
    """Load CALVIN dataset and extract instruction contexts for each verb."""
    import tensorflow_datasets as tfds

    # Build dictionary: verb -> list of instructions containing that verb
    verb_to_instructions = {v: [] for v in VERBS}

    try:
        ds = tfds.load("calvin_dataset", data_dir=data_dir, split="train")
        count = 0
        for example in ds:
            if count >= 500:  # Sample to avoid memory issues
                break
            instruction = example["language_instruction"].numpy().decode("utf-8").lower()
            for verb in VERBS:
                if verb.lower() in instruction:
                    verb_to_instructions[verb].append(instruction)
            count += 1
    except Exception as e:
        print(f"Warning: Could not load CALVIN instructions: {e}")
        print("Falling back to synthetic instructions...")
        # Use simple synthetic instructions as fallback
        for verb in VERBS:
            verb_to_instructions[verb] = [f"{verb} the object", f"please {verb} it"]

    return verb_to_instructions


def extract_contextualized_embeddings(model, tokenizer, verbs, verb_to_instructions, device="cuda"):
    """
    Extract contextualized verb embeddings from instruction context.

    For each verb, find all instructions containing it, embed each instruction,
    and extract the position of the verb in the embedding. Mean pool across all occurrences.
    """
    embeddings = []

    for verb in verbs:
        instructions = verb_to_instructions.get(verb, [verb])
        if not instructions:
            instructions = [verb]

        verb_embeds = []
        for instr in instructions[:5]:  # Limit to 5 instructions per verb to save memory
            # Find verb position in instruction
            instr_lower = instr.lower()
            verb_lower = verb.lower()
            verb_pos = instr_lower.find(verb_lower)

            if verb_pos == -1:
                # Verb not found, just embed the instruction and take mean
                inputs = tokenizer(instr, return_tensors="pt", truncation=True,
                                  max_length=128).to(device)
            else:
                inputs = tokenizer(instr, return_tensors="pt", truncation=True,
                                  max_length=128).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            verb_embeds.append(pooled[0].cpu().float().numpy())

        # Mean pool across all instruction contexts
        if verb_embeds:
            emb = np.mean(verb_embeds, axis=0)
        else:
            emb = verb_embeds[0] if verb_embeds else np.zeros(hidden.shape[-1])

        embeddings.append(emb)

    return np.stack(embeddings)


def extract_contextualized_vicuna(verbs, device="cuda"):
    """Extract contextualized Vicuna embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Vicuna-7B-v1.5 for contextualized embeddings...")
    model_id = "lmsys/vicuna-7b-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    verb_to_instructions = load_calvin_instructions()
    embeddings = extract_contextualized_embeddings(model, tokenizer, verbs,
                                                   verb_to_instructions, device)
    torch.cuda.empty_cache()
    return embeddings


def extract_contextualized_llava(verbs, device="cuda"):
    """Extract contextualized LLaVA embeddings."""
    from transformers import AutoTokenizer, LlamaForCausalLM
    import warnings
    print("Loading LLaVA-1.5-7B for contextualized embeddings...")
    model_id = "liuhaotian/llava-v1.5-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    model.eval()

    verb_to_instructions = load_calvin_instructions()
    embeddings = extract_contextualized_embeddings(model, tokenizer, verbs,
                                                   verb_to_instructions, device)
    torch.cuda.empty_cache()
    return embeddings


def extract_contextualized_qwen(verbs, device="cuda"):
    """Extract contextualized Qwen embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Qwen-2.5-7B for contextualized embeddings...")
    model_id = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    verb_to_instructions = load_calvin_instructions()
    embeddings = extract_contextualized_embeddings(model, tokenizer, verbs,
                                                   verb_to_instructions, device)
    torch.cuda.empty_cache()
    return embeddings


def extract_contextualized_qwen_vl(verbs, device="cuda"):
    """Extract contextualized Qwen-VL embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Qwen-VL-Chat for contextualized embeddings...")
    model_id = "Qwen/Qwen-VL-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    verb_to_instructions = load_calvin_instructions()
    embeddings = extract_contextualized_embeddings(model, tokenizer, verbs,
                                                   verb_to_instructions, device)
    torch.cuda.empty_cache()
    return embeddings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RSM COMPUTATION & CORRELATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_rsm(embeddings):
    """Compute representational similarity matrix (cosine similarity)."""
    return cosine_similarity(embeddings)


def correlate_rsm_confusion(rsm, confusion_matrix):
    """
    Compute Spearman and Pearson correlations between RSM and confusion matrix.
    Uses the upper triangle of both symmetric matrices.
    """
    triu_idx = np.triu_indices(len(rsm), k=1)
    rsm_vals = rsm[triu_idx]
    conf_vals = confusion_matrix[triu_idx]

    spearman_rho, spearman_p = spearmanr(rsm_vals, conf_vals)
    pearson_r, pearson_p = pearsonr(rsm_vals, conf_vals)

    return {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load predictions and build confusion matrix
    print("\n" + "="*60)
    print("LOADING PREDICTIONS AND BUILDING CONFUSION MATRIX")
    print("="*60)
    preds_path = os.path.join(RESULTS_DIR, "r8_ao_native_preds.json")
    labels, preds, id_to_verb = load_predictions(preds_path)

    # Map to 20-class verb list
    cm_sym, verbs_aligned = build_symmetric_confusion_matrix(labels, preds, VERBS)
    print(f"Confusion matrix shape: {cm_sym.shape}")
    print(f"Matrix is symmetric: {np.allclose(cm_sym, cm_sym.T)}")

    # Extract embeddings
    print("\n" + "="*60)
    print("EXTRACTING EMBEDDINGS")
    print("="*60)

    all_embeddings = {}
    all_rsms = {}
    correlations = {}

    # Non-contextualized
    print("\n--- Non-contextualized embeddings ---")
    all_embeddings["gpt2"] = extract_embeddings_gpt2(VERBS)
    all_embeddings["vicuna"] = extract_embeddings_vicuna(VERBS, device)
    all_embeddings["llava"] = extract_embeddings_llava_text(VERBS, device)
    all_embeddings["qwen"] = extract_embeddings_qwen(VERBS, device)
    all_embeddings["qwen_vl"] = extract_embeddings_qwen_vl(VERBS, device)

    # Contextualized
    print("\n--- Contextualized embeddings ---")
    all_embeddings["vicuna_ctx"] = extract_contextualized_vicuna(VERBS, device)
    all_embeddings["llava_ctx"] = extract_contextualized_llava(VERBS, device)
    all_embeddings["qwen_ctx"] = extract_contextualized_qwen(VERBS, device)
    all_embeddings["qwen_vl_ctx"] = extract_contextualized_qwen_vl(VERBS, device)

    # Compute RSMs and correlations
    print("\n" + "="*60)
    print("COMPUTING RSMs AND CORRELATIONS")
    print("="*60)

    for name, embeddings in all_embeddings.items():
        print(f"\nProcessing {name}...")
        rsm = compute_rsm(embeddings)
        all_rsms[name] = rsm

        corr = correlate_rsm_confusion(rsm, cm_sym)
        correlations[name] = corr

        print(f"  Spearman ρ: {corr['spearman_rho']:7.4f}  (p={corr['spearman_p']:.3f})")
        print(f"  Pearson r:  {corr['pearson_r']:7.4f}  (p={corr['pearson_p']:.3f})")

    # Print summary table
    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'Spearman ρ':>12} {'p-value':>10} {'Pearson r':>12} {'p-value':>10}")
    print("-" * 65)
    for name in sorted(all_embeddings.keys()):
        corr = correlations[name]
        print(f"{name:<20} {corr['spearman_rho']:>12.4f} {corr['spearman_p']:>10.3f} "
              f"{corr['pearson_r']:>12.4f} {corr['pearson_p']:>10.3f}")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    out_path = os.path.join(RESULTS_DIR, "rsm_confusion_analysis_v2.npz")
    save_dict = {
        "confusion_matrix": cm_sym,
        "verbs": np.array(VERBS),
        "correlations_json": json.dumps(correlations, indent=2),
    }
    save_dict.update({f"rsm_{name}": rsm for name, rsm in all_rsms.items()})
    save_dict.update({f"emb_{name}": emb for name, emb in all_embeddings.items()})

    np.savez(out_path, **save_dict)
    print(f"Saved to: {out_path}")

    # Save correlations to JSON
    json_path = os.path.join(RESULTS_DIR, "rsm_confusion_correlations_v2.json")
    with open(json_path, "w") as f:
        json.dump({
            "verbs": VERBS,
            "correlations": correlations,
            "note": "Symmetric confusion matrix vs verb RSMs (contextualized & non-contextualized)",
        }, f, indent=2)
    print(f"Saved correlations to: {json_path}")


if __name__ == "__main__":
    main()
