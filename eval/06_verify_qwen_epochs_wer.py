"""
Script: 06_verify_qwen_epochs_wer.py

Evaluates the Qwen LoRA models trained WITHOUT inline confidence scores.
This is now apples-to-apples with:

    reproducible/09_verify_qwen_confidence_epochs_wer.py
    reproducible/11_verify_qwen_placebo_confidence_epochs_wer.py

All three use the same char-level IPA acoustic frontend:

    models/phonemic_char_ipa_FULL_best.pt
    data/char_ipa_vocab.json

Run from project root:
    python reproducible/06_verify_qwen_epochs_wer.py

Useful smoke test:
    python reproducible/06_verify_qwen_epochs_wer.py --max-samples 50

Files needed:
    models/phonemic_char_ipa_FULL_best.pt          - char-IPA acoustic model
    data/char_ipa_vocab.json                       - char-level IPA vocabulary
    lora_no_confidence_models/checkpoint-2102/     - no-confidence LoRA Epoch 1
    lora_no_confidence_models/checkpoint-4204/     - no-confidence LoRA Epoch 2
    lora_no_confidence_models/checkpoint-6306/     - no-confidence LoRA Epoch 3
    data/LibriSpeech/test-clean/                   - downloaded automatically if absent
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

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


CTC_WEIGHTS = "models/phonemic_char_ipa_FULL_best.pt"
VOCAB_PATH = "data/char_ipa_vocab.json"
DATA_ROOT = "./data"
LIBRI_ROOT = "./data/LibriSpeech/test-clean"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_CSV = "results/qwen_no_confidence_char_ipa_epoch_wer.csv"

SYSTEM_PROMPT = "You are an expert phonetic decoder. Convert the following IPA string back into standard English."

CHECKPOINTS = {
    "Ep 1 (ckpt-2102)": "lora_no_confidence_models/checkpoint-2102",
    "Ep 2 (ckpt-4204)": "lora_no_confidence_models/checkpoint-4204",
    "Ep 3 (ckpt-6306)": "lora_no_confidence_models/checkpoint-6306",
}


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


def bootstrap_ci(errors, lengths, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(errors)
    wers = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        e = np.sum(errors[idx])
        l = np.sum(lengths[idx])
        wers.append(e / l if l > 0 else 0)
    wers.sort()
    return wers[25] * 100, wers[975] * 100


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


def build_ipa_dataset(pairs, ctc_model, ipa_tokenizer, device):
    dataset = []
    for audio_path, ref in tqdm(pairs, desc="CTC char-IPA decoding"):
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        true_audio_len_s = len(audio) / 16000.0
        audio = openai_whisper.pad_or_trim(audio.astype(np.float32))
        mel = openai_whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = ctc_model(mel)[0, : int(true_audio_len_s * 50), :]
        dataset.append((ref, greedy_char_ipa(logits, ipa_tokenizer)))
    return dataset


def load_base_model():
    if torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
    return AutoModelForCausalLM.from_pretrained(QWEN_MODEL)


def evaluate_checkpoint(ckpt_path, tokenizer, ipa_dataset, batch_size):
    # PEFT mutates the supplied base model; use a clean base per checkpoint.
    base_model = load_base_model()
    llm = PeftModel.from_pretrained(base_model, ckpt_path)
    llm.eval()

    errors, lengths = [], []
    refs = [ref for ref, _ in ipa_dataset]
    ipas = [ipa for _, ipa in ipa_dataset]

    for start in tqdm(range(0, len(ipa_dataset), batch_size), desc=os.path.basename(ckpt_path)):
        batch_ipas = ipas[start : start + batch_size]
        batch_refs = refs[start : start + batch_size]
        prompts = [
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{ipa}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for ipa in batch_ipas
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(llm.device)
        with torch.no_grad():
            out = llm.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j, ref in enumerate(batch_refs):
            in_len = inputs.input_ids[j].size(0)
            hyp = norm(tokenizer.decode(out[j][in_len:], skip_special_tokens=True))
            result = jiwer.process_words(norm(ref), hyp)
            errors.append(result.substitutions + result.deletions + result.insertions)
            lengths.append(len(norm(ref).split()))

    errors = np.array(errors)
    lengths = np.array(lengths)
    wer = np.sum(errors) / np.sum(lengths) * 100
    ci_lo, ci_hi = bootstrap_ci(errors, lengths)
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return wer, ci_lo, ci_hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test-clean utterances for a smoke test.")
    parser.add_argument("--batch-size", type=int, default=16, help="Qwen generation batch size.")
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    args = parser.parse_args()

    for path in [CTC_WEIGHTS, VOCAB_PATH]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            sys.exit(1)
    for ckpt in CHECKPOINTS.values():
        if not os.path.exists(ckpt):
            print(f"[ERROR] Missing checkpoint: {ckpt}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("Qwen NO-confidence LoRA evaluation (char-IPA apples-to-apples)")
    print(f"Device: {device}")
    print(f"Acoustic model: {CTC_WEIGHTS}")
    print(f"IPA vocab: {VOCAB_PATH}")
    print("=" * 72)

    import torchaudio

    torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    pairs = load_pairs(LIBRI_ROOT)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    print(f"Loaded {len(pairs)} test-clean pairs.")

    ipa_tokenizer = CharIPATokenizer()
    ipa_tokenizer.load(VOCAB_PATH)
    ctc_model = WhisperForPhonemicCTC(ipa_tokenizer.vocab_size).to(device)
    ctc_model.load_state_dict(torch.load(CTC_WEIGHTS, map_location=device, weights_only=False))
    ctc_model.eval()

    ipa_dataset = build_ipa_dataset(pairs, ctc_model, ipa_tokenizer, device)
    del ctc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    print("\n" + "-" * 72)
    for label, ckpt in CHECKPOINTS.items():
        epoch_label = label.split()[0] + " " + label.split()[1]
        print(f"\nEvaluating {label}: {ckpt}")
        wer, ci_lo, ci_hi = evaluate_checkpoint(ckpt, tokenizer, ipa_dataset, args.batch_size)
        rows.append(
            {
                "epoch": epoch_label,
                "no_confidence_checkpoint": ckpt,
                "n": len(ipa_dataset),
                "wer": wer,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "acoustic_model": CTC_WEIGHTS,
                "ipa_vocab": VOCAB_PATH,
            }
        )
        print(f"  NO confidence: {wer:.2f}%  CI [{ci_lo:.2f}, {ci_hi:.2f}]")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("Summary")
    print("Epoch | No-confidence WER")
    for row in rows:
        print(f"{row['epoch']:>5} | {row['wer']:.2f}% [{row['ci_low']:.2f}, {row['ci_high']:.2f}]")
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
