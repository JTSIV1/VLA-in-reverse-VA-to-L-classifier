"""
Compute per-sentence log-likelihood of CALVIN instructions under each
autoregressive language model.

Hypothesis: if VLMs have similar log-likelihood to their LM backbones on
CALVIN text but higher RSM correlation with motor similarity, then it's the
visual grounding (not text familiarity) that aligns verb representations.

Models (autoregressive, produce token-level log-probs):
  - GPT-2 (117M)
  - Vicuna-7B-v1.5 (LLaVA's LM backbone)
  - LLaVA-1.5-7B (Vicuna + visual training)
  - Qwen-VL-Chat (Qwen + visual training)

Outputs:
  results/instruction_loglikelihoods.json — per-model, per-instruction stats
  results/instruction_loglikelihoods.npz  — raw arrays
"""

import os
import sys
import json
import numpy as np
import torch
import pandas as pd

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HF_HOME = "/data/user_data/wenjiel2/.cache/huggingface"
os.environ.setdefault("HF_HOME", HF_HOME)


def compute_loglikelihood(model, tokenizer, text, device="cuda"):
    """
    Compute total and mean per-token log-likelihood of `text` under `model`.

    Returns (total_ll, mean_ll, n_tokens).
    total_ll  = sum of log p(t_i | t_<i)
    mean_ll   = total_ll / n_tokens
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is cross-entropy averaged over tokens
        # CE = -mean(log p(t_i | t_<i))
        n_tokens = input_ids.shape[1] - 1  # first token has no prediction
        total_ll = -outputs.loss.item() * n_tokens
        mean_ll = -outputs.loss.item()

    return total_ll, mean_ll, n_tokens


def load_model(model_key):
    """Load model and tokenizer. Returns (name, model, tokenizer, device)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    configs = {
        "gpt2": ("gpt2", {}),
        "vicuna": ("lmsys/vicuna-7b-v1.5", {"torch_dtype": torch.float16, "device_map": "auto"}),
        "llava": ("liuhaotian/llava-v1.5-7b", {"torch_dtype": torch.float16, "device_map": "auto"}),
        "qwen-vl": ("Qwen/Qwen-VL-Chat", {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}),
    }

    model_id, kwargs = configs[model_key]
    trust_remote = kwargs.pop("trust_remote_code", False)

    print("Loading {} ...".format(model_id))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_key == "llava":
        # LLaVA-1.5-7b checkpoint has full Llama/Vicuna weights merged in
        from transformers import LlamaForCausalLM
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LlamaForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote, **kwargs)
        model.eval()

    device = next(model.parameters()).device
    return model_key, model, tokenizer, device


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["gpt2", "vicuna", "llava", "qwen-vl"],
                        choices=["gpt2", "vicuna", "llava", "qwen-vl"])
    args = parser.parse_args()

    # Load unique CALVIN instructions
    csv_path = os.path.join(PROJ_ROOT, "calvin_train_annotations.csv")
    df = pd.read_csv(csv_path)
    instructions = sorted(df["instruction"].unique().tolist())
    print("Loaded {} unique instructions".format(len(instructions)))

    results = {}
    arrays = {}

    for model_key in args.models:
        print("\n" + "=" * 60)
        print("Model: {}".format(model_key))
        print("=" * 60)

        name, model, tokenizer, device = load_model(model_key)

        total_lls = []
        mean_lls = []
        n_tokens_list = []

        for i, instr in enumerate(instructions):
            total_ll, mean_ll, n_tok = compute_loglikelihood(model, tokenizer, instr, device)
            total_lls.append(total_ll)
            mean_lls.append(mean_ll)
            n_tokens_list.append(n_tok)

            if i < 5 or (i + 1) % 50 == 0:
                print("  [{:3d}/{}] {:50s}  mean_ll={:.3f}  n_tok={}".format(
                    i + 1, len(instructions), instr[:50], mean_ll, n_tok))

        total_lls = np.array(total_lls)
        mean_lls = np.array(mean_lls)
        n_tokens_arr = np.array(n_tokens_list)

        results[model_key] = {
            "mean_ll_avg": float(mean_lls.mean()),
            "mean_ll_std": float(mean_lls.std()),
            "total_ll_avg": float(total_lls.mean()),
            "perplexity": float(np.exp(-mean_lls.mean())),
            "n_instructions": len(instructions),
            "avg_tokens": float(n_tokens_arr.mean()),
        }
        arrays["{}_mean_ll".format(model_key)] = mean_lls
        arrays["{}_total_ll".format(model_key)] = total_lls
        arrays["{}_n_tokens".format(model_key)] = n_tokens_arr

        print("\nSummary for {}:".format(model_key))
        print("  Mean LL (avg): {:.4f} +/- {:.4f}".format(mean_lls.mean(), mean_lls.std()))
        print("  Perplexity:    {:.2f}".format(np.exp(-mean_lls.mean())))
        print("  Avg tokens:    {:.1f}".format(n_tokens_arr.mean()))

        # Free GPU memory
        if model_key != "gpt2":
            import gc
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print("  (freed GPU memory)")

    # Save
    arrays["instructions"] = np.array(instructions)

    json_path = os.path.join(RESULTS_DIR, "instruction_loglikelihoods.json")
    npz_path = os.path.join(RESULTS_DIR, "instruction_loglikelihoods.npz")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    np.savez(npz_path, **arrays)

    print("\n" + "=" * 60)
    print("Saved to:")
    print("  {}".format(json_path))
    print("  {}".format(npz_path))

    # Comparison table
    print("\n── Log-likelihood comparison ──")
    print("{:15s} {:>12s} {:>12s} {:>10s}".format(
        "Model", "Mean LL", "Perplexity", "Avg Tok"))
    for mk in args.models:
        r = results[mk]
        print("{:15s} {:12.4f} {:12.2f} {:10.1f}".format(
            mk, r["mean_ll_avg"], r["perplexity"], r["avg_tokens"]))


if __name__ == "__main__":
    main()
