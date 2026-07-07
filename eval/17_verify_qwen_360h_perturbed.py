"""
evaluate_qwen_inference_tuning.py

Evaluates the perturbed Qwen-360 model (lora_adapter_360_perturbed_unsloth/lora_adapter_360_perturbed_unsloth)
on LibriSpeech test-clean using:
1. Baseline: Standard decoding (no hacks)
2. Method A: Repetition Penalty (repetition_penalty = 1.1)
3. Method B (0.4): Minimum Length Constraint (min_new_tokens = ipa_word_count * 0.4)
4. Method B (0.7): Minimum Length Constraint (min_new_tokens = ipa_word_count * 0.7)
5. Combined (A + B 0.4): Rep Penalty = 1.1, Min Length = 0.4
6. Combined (A + B 0.7): Rep Penalty = 1.1, Min Length = 0.7
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import whisper as openai_whisper
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath("."))

# File paths
CTC_WEIGHTS = "models/phonemic_char_ipa_FULL_best.pt"
VOCAB_PATH = "data/char_ipa_vocab.json"
DATA_ROOT = "./data"
LIBRI_ROOT = "./data/LibriSpeech/test-clean"
QWEN_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_ADAPTER = "GITHUB_UPLOAD/models/lora_adapter_360_perturbed_unsloth/lora_adapter_360_perturbed_unsloth"

SYSTEM_PROMPT = "You are an expert phonetic decoder. Convert the following IPA string back into standard English."

class CharIPATokenizer:
    def __init__(self):
        self.blank_id = 0
        self.pad_id = 1
        self.unk_id = 2
        self.char2id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2}
        self.id2char = {0: "<BLANK>", 1: "<PAD>", 2: "<UNK>"}
        self.vocab_size = 3

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char2id = data["char2id"]
        self.id2char = {int(v): k for k, v in self.char2id.items()}
        self.vocab_size = len(self.char2id)

class ConvCTCAdapter(nn.Module):
    def __init__(self, d_model, vocab_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.ln1(x).transpose(1, 2)
        x = self.drop1(self.act1(self.conv1(x))).transpose(1, 2)
        x = self.ln2(x).transpose(1, 2)
        x = self.drop2(self.act2(self.conv2(x))).transpose(1, 2)
        return self.linear(self.ln3(x))

class WhisperForPhonemicCTC(nn.Module):
    def __init__(self, vocab_size, dropout=0.1):
        super().__init__()
        full = openai_whisper.load_model("tiny")
        self.encoder = full.encoder
        self.dims = full.dims
        self.ctc_head = ConvCTCAdapter(self.dims.n_audio_state, vocab_size, dropout=dropout)
        del full

    def forward(self, mel):
        return self.ctc_head(self.encoder(mel))

def load_pairs(root):
    pairs = []
    for trans_file in glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True):
        folder = os.path.dirname(trans_file)
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uid, *parts = line.split()
                flac = os.path.join(folder, uid + ".flac")
                if os.path.exists(flac):
                    pairs.append((flac, " ".join(parts).lower()))
    return pairs

def norm(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower().translate(str.maketrans("", "", ".,?!;:\"'")),
    ).strip()

def greedy_char_ipa(logits, tokenizer):
    probs = torch.softmax(logits, dim=-1)
    ids = torch.argmax(probs, dim=-1).tolist()
    decoded = []
    prev = None
    space_id = tokenizer.char2id.get("<SPACE>")

    for tid in ids:
        if tid == tokenizer.blank_id:
            prev = tid
            continue
        if tid == prev:
            continue
        if tid == space_id:
            decoded.append(" ")
        else:
            decoded.append(tokenizer.id2char.get(tid, ""))
        prev = tid

    return re.sub(r" +", " ", "".join(decoded)).strip()

def evaluate_qwen_tuning(ipa_dataset, qwen_model, tokenizer, config_name, rep_penalty, use_min_len, min_len_ratio, batch_size):
    hyps = []
    refs = [ref for ref, _, _ in ipa_dataset]
    
    # Process batch-by-batch
    for start in tqdm(range(0, len(ipa_dataset), batch_size), desc=f"Evaluating {config_name}"):
        batch = ipa_dataset[start : start + batch_size]
        prompts = [
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{ipa}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for _, ipa, _ in batch
        ]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(qwen_model.device)
        
        # Generation arguments
        gen_kwargs = {
            "max_new_tokens": 256,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if rep_penalty is not None:
            gen_kwargs["repetition_penalty"] = rep_penalty
            
        if use_min_len:
            min_ipa_words = min(w_count for _, _, w_count in batch)
            gen_kwargs["min_new_tokens"] = max(1, int(min_ipa_words * min_len_ratio))

        with torch.no_grad():
            out = qwen_model.generate(**inputs, **gen_kwargs)
            
        for j in range(len(batch)):
            in_len = inputs.input_ids[j].size(0)
            hyp = norm(tokenizer.decode(out[j][in_len:], skip_special_tokens=True))
            hyps.append(hyp)
            
    # Compute WER and per-sentence error arrays
    errors, lengths = [], []
    for r, h in zip(refs, hyps):
        res = jiwer.process_words(norm(r), h)
        errors.append(res.substitutions + res.deletions + res.insertions)
        lengths.append(len(norm(r).split()))

    errors_arr = np.array(errors, dtype=np.float64)
    lengths_arr = np.array(lengths, dtype=np.float64)
    wer = np.sum(errors_arr) / np.sum(lengths_arr) * 100

    # 1,000-iteration sentence-level bootstrap resampling for 95% CI
    rng = np.random.default_rng(seed=42)
    n = len(errors_arr)
    boot_wers = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        boot_wers.append(np.sum(errors_arr[idx]) / np.sum(lengths_arr[idx]) * 100)
    ci_low  = float(np.percentile(boot_wers, 2.5))
    ci_high = float(np.percentile(boot_wers, 97.5))

    return wer, ci_low, ci_high, hyps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick checks.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio dataset
    import torchaudio
    torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    pairs = load_pairs(LIBRI_ROOT)
    if args.max_samples is not None:
        pairs = pairs[:args.max_samples]
    print(f"Loaded {len(pairs)} test-clean FLAC files.")

    # 1. Run Acoustic CTC Model to pre-compute IPA predictions
    print("Loading Acoustic CTC model...")
    ipa_tokenizer = CharIPATokenizer()
    ipa_tokenizer.load(VOCAB_PATH)
    ctc_model = WhisperForPhonemicCTC(ipa_tokenizer.vocab_size).to(device)
    ctc_model.load_state_dict(torch.load(CTC_WEIGHTS, map_location=device, weights_only=False))
    ctc_model.eval()

    print("Acoustic decoding to IPA...")
    raw_dataset = []
    for audio_path, ref in tqdm(pairs, desc="Acoustic CTC"):
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        true_audio_len_s = len(audio) / 16000.0
        audio = openai_whisper.pad_or_trim(audio.astype(np.float32))
        mel = openai_whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = ctc_model(mel)[0, : int(true_audio_len_s * 50), :]
        ipa_str = greedy_char_ipa(logits, ipa_tokenizer)
        word_count = len(ipa_str.split())
        raw_dataset.append((ref, ipa_str, word_count))

    del ctc_model
    torch.cuda.empty_cache()

    # Sort dataset by IPA word count (bucket batching for maximum speed and optimal min_new_tokens accuracy)
    raw_dataset.sort(key=lambda x: x[2])

    # 2. Load Qwen Model
    print(f"\nLoading perturbed Qwen-360 model from: {QWEN_ADAPTER}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_BASE,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa" if torch.cuda.is_available() else None,
    )
    qwen_model = PeftModel.from_pretrained(base_qwen, QWEN_ADAPTER)
    qwen_model.eval()

    # 0. Baseline (No hacks)
    wer_base, ci_base_lo, ci_base_hi, _ = evaluate_qwen_tuning(
        raw_dataset, qwen_model, tokenizer,
        config_name="Baseline (No Hacks)",
        rep_penalty=None, use_min_len=False, min_len_ratio=0.0, batch_size=args.batch_size
    )

    # A. Repetition Penalty only (1.1)
    wer_a, ci_a_lo, ci_a_hi, _ = evaluate_qwen_tuning(
        raw_dataset, qwen_model, tokenizer,
        config_name="Method A (Rep Penalty = 1.10)",
        rep_penalty=1.10, use_min_len=False, min_len_ratio=0.0, batch_size=args.batch_size
    )
    
    # B. Min Length Constraint (0.4)
    wer_b4, ci_b4_lo, ci_b4_hi, _ = evaluate_qwen_tuning(
        raw_dataset, qwen_model, tokenizer,
        config_name="Method B (Min Length = 0.4)",
        rep_penalty=None, use_min_len=True, min_len_ratio=0.4, batch_size=args.batch_size
    )

    # B. Min Length Constraint (0.7)
    wer_b7, ci_b7_lo, ci_b7_hi, _ = evaluate_qwen_tuning(
        raw_dataset, qwen_model, tokenizer,
        config_name="Method B (Min Length = 0.7)",
        rep_penalty=None, use_min_len=True, min_len_ratio=0.7, batch_size=args.batch_size
    )

    # C. Combined (Rep Penalty 1.10 + Min Length 0.4)
    wer_c4, ci_c4_lo, ci_c4_hi, _ = evaluate_qwen_tuning(
        raw_dataset, qwen_model, tokenizer,
        config_name="Method A + B (Rep 1.10, Min 0.4)",
        rep_penalty=1.10, use_min_len=True, min_len_ratio=0.4, batch_size=args.batch_size
    )

    # C. Combined (Rep Penalty 1.10 + Min Length 0.7)
    wer_c7, ci_c7_lo, ci_c7_hi, _ = evaluate_qwen_tuning(
        raw_dataset, qwen_model, tokenizer,
        config_name="Method A + B (Rep 1.10, Min 0.7)",
        rep_penalty=1.10, use_min_len=True, min_len_ratio=0.7, batch_size=args.batch_size
    )

    print("\n" + "=" * 80)
    print("INFERENCE DECIDERS EVALUATION SUMMARY (PERTURBED TRAINING MODEL)")
    print("-" * 80)
    print(f"Dataset                           : LibriSpeech test-clean (N={len(pairs)})")
    print(f"Baseline (No Hacks) WER           : {wer_base:.2f}%  95% CI [{ci_base_lo:.2f}, {ci_base_hi:.2f}]")
    print(f"Method A (Rep Penalty 1.10) WER   : {wer_a:.2f}%  95% CI [{ci_a_lo:.2f}, {ci_a_hi:.2f}]")
    print(f"Method B (Min Length 0.4) WER     : {wer_b4:.2f}%  95% CI [{ci_b4_lo:.2f}, {ci_b4_hi:.2f}]")
    print(f"Method B (Min Length 0.7) WER     : {wer_b7:.2f}%  95% CI [{ci_b7_lo:.2f}, {ci_b7_hi:.2f}]")
    print(f"Combined (Rep 1.10 + Min 0.4) WER : {wer_c4:.2f}%  95% CI [{ci_c4_lo:.2f}, {ci_c4_hi:.2f}]")
    print(f"Combined (Rep 1.10 + Min 0.7) WER : {wer_c7:.2f}%  95% CI [{ci_c7_lo:.2f}, {ci_c7_hi:.2f}]")
    print("="  * 80)
    print("NOTE: CIs use 1,000-iteration sentence-level bootstrap resampling (seed=42).")
    print("NOTE: Same N=2,620 test-clean split; same jiwer WER computation as main eval.")

if __name__ == "__main__":
    main()
