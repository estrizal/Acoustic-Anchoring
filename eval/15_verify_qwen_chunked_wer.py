"""
Evaluate Qwen LoRA ASR correctors with the chunked char-IPA frontend.

This script is the apples-to-apples replacement for the older single-window
WER scripts when utterances can exceed Whisper's 30 s encoder window.

Examples:
    python reproducible/15_verify_qwen_chunked_wer.py --mode confidence
    python reproducible/15_verify_qwen_chunked_wer.py --mode placebo --max-samples 50
    python reproducible/15_verify_qwen_chunked_wer.py --mode no_confidence --only-long
    python reproducible/15_verify_qwen_chunked_wer.py --mode placebo --checkpoint models/lora_placebo_confidence_models/checkpoint-12591 --single-window --prompt-chunk-words 25
    python reproducible/15_verify_qwen_chunked_wer.py --mode word_boundary --checkpoint path/to/checkpoint
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
import torchaudio
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from chunked_char_ipa_frontend import (  # noqa: E402
    decode_audio_to_words,
    load_frontend,
    words_to_boundary_prompt,
    words_to_confidence_prompt,
    words_to_placebo_prompt,
    words_to_plain_ipa,
)


CTC_WEIGHTS = "models/phonemic_char_ipa_FULL_best.pt"
VOCAB_PATH = "data/char_ipa_vocab.json"
DATA_ROOT = "./data"
LIBRI_ROOT = "./data/LibriSpeech/test-clean"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPTS = {
    "no_confidence": "You are an expert phonetic decoder. Convert the following IPA string back into standard English.",
    "confidence": (
        "You are an expert phonetic decoder. Your task is to decode noisy IPA "
        "phonetic transcripts into standard English. To help you, each word's "
        "phonetic transcription is preceded by a confidence score in brackets, like [0.94]."
    ),
    "placebo": (
        "You are an expert phonetic decoder. Your task is to decode noisy IPA "
        "phonetic transcripts into standard English. To help you, each word's "
        "phonetic transcription is preceded by a confidence score in brackets, like [0.94]."
    ),
    "word_boundary": (
        "You are an expert phonetic decoder. Convert the following word-bounded "
        "IPA string back into standard English."
    ),
}

DEFAULT_CHECKPOINTS = {
    "no_confidence": {
        "Ep 3": "models/lora_no_confidence_models/checkpoint-6306",
    },
    "confidence": {
        "Ep 3": "models/lora_yes_confidence_models/checkpoint-12591",
    },
    "placebo": {
        "Ep 3": "models/lora_placebo_confidence_models/checkpoint-12591",
    },
    "word_boundary": {
        "Ep 3": "models/lora_word_boundary/checkpoint-12612",
    },
}


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
                    pairs.append((uid, flac, " ".join(parts).lower()))
    return pairs


def norm(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower().translate(str.maketrans("", "", ".,?!;:\"'")),
    ).strip()


def duration_s(path):
    info = sf.info(path)
    return info.frames / info.samplerate


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


def words_to_prompt(words, mode):
    if mode == "no_confidence":
        return words_to_plain_ipa(words)
    if mode == "confidence":
        return words_to_confidence_prompt(words)
    if mode == "placebo":
        return words_to_placebo_prompt(words)
    if mode == "word_boundary":
        return words_to_boundary_prompt(words)
    raise ValueError(mode)


def build_prompt_dataset(pairs, ctc_model, ipa_tokenizer, device, mode, chunked):
    dataset = []
    for uid, audio_path, ref in tqdm(pairs, desc=f"CTC char-IPA decoding ({'chunked' if chunked else 'single'})"):
        words = decode_audio_to_words(
            audio_path,
            ctc_model,
            ipa_tokenizer,
            device,
            chunked=chunked,
        )
        dataset.append((uid, ref, words_to_prompt(words, mode)))
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


def prompt_units(prompt, mode):
    parts = prompt.split()
    if mode in {"confidence", "placebo"}:
        units = []
        i = 0
        while i < len(parts):
            if parts[i].startswith("[") and parts[i].endswith("]") and i + 1 < len(parts):
                units.append(" ".join(parts[i : i + 2]))
                i += 2
            else:
                units.append(parts[i])
                i += 1
    elif mode == "word_boundary":
        units = []
        i = 0
        while i < len(parts):
            if parts[i] == "<w>" and i + 2 < len(parts) and parts[i + 2] == "</w>":
                units.append(" ".join(parts[i : i + 3]))
                i += 3
            else:
                units.append(parts[i])
                i += 1
    else:
        units = parts
    return units


def split_prompt_chunks(prompt, mode, chunk_words, min_words):
    if not chunk_words or chunk_words <= 0:
        return [prompt]

    units = prompt_units(prompt, mode)
    if min_words and len(units) <= min_words:
        return [prompt]

    chunks = []

    for start in range(0, len(units), chunk_words):
        chunk = " ".join(units[start : start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [prompt]


def merge_chunk_hypotheses(chunk_hyps):
    return norm(" ".join(h for h in chunk_hyps if h.strip()))


def generate_prompt_batches(llm, tokenizer, prompts, system_prompt, batch_size, desc, max_new_tokens):
    outputs = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        chat_prompts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for prompt in batch_prompts
        ]
        inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(llm.device)
        with torch.no_grad():
            out = llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j in range(len(batch_prompts)):
            in_len = inputs.input_ids[j].size(0)
            outputs.append(norm(tokenizer.decode(out[j][in_len:], skip_special_tokens=True)))
    return outputs


def evaluate_checkpoint(
    ckpt_path,
    tokenizer,
    prompt_dataset,
    system_prompt,
    batch_size,
    mode,
    prompt_chunk_words,
    prompt_chunk_min_words,
    max_new_tokens,
):
    base_model = load_base_model()
    llm = PeftModel.from_pretrained(base_model, ckpt_path)
    llm.eval()

    errors, lengths = [], []
    rows = []

    if prompt_chunk_words and prompt_chunk_words > 0:
        flat_prompts = []
        spans = []
        for uid, ref, prompt in prompt_dataset:
            start = len(flat_prompts)
            chunks = split_prompt_chunks(prompt, mode, prompt_chunk_words, prompt_chunk_min_words)
            flat_prompts.extend(chunks)
            spans.append((uid, ref, prompt, start, len(flat_prompts), len(chunks)))

        flat_hyps = generate_prompt_batches(
            llm,
            tokenizer,
            flat_prompts,
            system_prompt,
            batch_size,
            desc=f"{os.path.basename(ckpt_path)} chunks",
            max_new_tokens=max_new_tokens,
        )
        for uid, ref, prompt, start, end, n_chunks in spans:
            hyp = merge_chunk_hypotheses(flat_hyps[start:end])
            result = jiwer.process_words(norm(ref), hyp)
            err = result.substitutions + result.deletions + result.insertions
            length = len(norm(ref).split())
            errors.append(err)
            lengths.append(length)
            rows.append(
                {
                    "utt_id": uid,
                    "ref": ref,
                    "hyp": hyp,
                    "prompt": prompt,
                    "errors": err,
                    "words": length,
                    "prompt_chunks": n_chunks,
                }
            )
    else:
        prompts = [prompt for _, _, prompt in prompt_dataset]
        hyps = generate_prompt_batches(
            llm,
            tokenizer,
            prompts,
            system_prompt,
            batch_size,
            desc=os.path.basename(ckpt_path),
            max_new_tokens=max_new_tokens,
        )
        for (uid, ref, prompt), hyp in zip(prompt_dataset, hyps):
            result = jiwer.process_words(norm(ref), hyp)
            err = result.substitutions + result.deletions + result.insertions
            length = len(norm(ref).split())
            errors.append(err)
            lengths.append(length)
            rows.append(
                {
                    "utt_id": uid,
                    "ref": ref,
                    "hyp": hyp,
                    "prompt": prompt,
                    "errors": err,
                    "words": length,
                    "prompt_chunks": 1,
                }
            )

    errors = np.array(errors)
    lengths = np.array(lengths)
    wer = np.sum(errors) / np.sum(lengths) * 100
    ci_lo, ci_hi = bootstrap_ci(errors, lengths)
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return wer, ci_lo, ci_hi, rows


def checkpoint_map(mode, checkpoint):
    if checkpoint:
        return {"custom": checkpoint}
    if mode not in DEFAULT_CHECKPOINTS:
        raise ValueError(f"No default checkpoints registered for mode={mode}; pass --checkpoint.")
    return DEFAULT_CHECKPOINTS[mode]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(SYSTEM_PROMPTS), required=True)
    parser.add_argument("--checkpoint", default=None, help="Evaluate one custom LoRA checkpoint.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--only-long", action="store_true", help="Evaluate only utterances longer than 30 s.")
    parser.add_argument("--single-window", action="store_true", help="Use old single-window truncating frontend.")
    parser.add_argument(
        "--prompt-chunk-words",
        type=int,
        default=0,
        help="Split each IPA prompt into N word-like units, decode chunks independently, then concatenate.",
    )
    parser.add_argument(
        "--prompt-chunk-min-words",
        type=int,
        default=0,
        help="Only prompt-chunk utterances above this many word-like IPA units.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--save-predictions", default=None)
    args = parser.parse_args()

    torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    pairs = load_pairs(LIBRI_ROOT)
    if args.only_long:
        pairs = [pair for pair in pairs if duration_s(pair[1]) > 30.0]
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]

    ckpts = checkpoint_map(args.mode, args.checkpoint)
    for path in [CTC_WEIGHTS, VOCAB_PATH, *ckpts.values()]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"Chunked Qwen WER evaluation: mode={args.mode}")
    print(f"Device: {device}")
    print(f"Frontend: {'single-window/truncating' if args.single_window else 'chunked overlap'}")
    print(f"Loaded {len(pairs)} test-clean pairs.")
    print("=" * 72)

    ctc_model, ipa_tokenizer = load_frontend(CTC_WEIGHTS, VOCAB_PATH, device)
    prompt_dataset = build_prompt_dataset(
        pairs,
        ctc_model,
        ipa_tokenizer,
        device,
        args.mode,
        chunked=not args.single_window,
    )
    del ctc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    csv_rows = []
    prediction_rows = []
    for label, ckpt in ckpts.items():
        print(f"\nEvaluating {label}: {ckpt}")
        wer, ci_lo, ci_hi, rows = evaluate_checkpoint(
            ckpt,
            tokenizer,
            prompt_dataset,
            SYSTEM_PROMPTS[args.mode],
            args.batch_size,
            args.mode,
            args.prompt_chunk_words,
            args.prompt_chunk_min_words,
            args.max_new_tokens,
        )
        csv_rows.append(
            {
                "mode": args.mode,
                "checkpoint_label": label,
                "checkpoint": ckpt,
                "frontend": "single_window" if args.single_window else "chunked",
                "prompt_chunk_words": args.prompt_chunk_words,
                "prompt_chunk_min_words": args.prompt_chunk_min_words,
                "n": len(prompt_dataset),
                "wer": wer,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "acoustic_model": CTC_WEIGHTS,
                "ipa_vocab": VOCAB_PATH,
            }
        )
        for row in rows:
            row.update({"checkpoint_label": label, "checkpoint": ckpt, "mode": args.mode})
            prediction_rows.append(row)
        print(f"  WER: {wer:.2f}%  CI [{ci_lo:.2f}, {ci_hi:.2f}]")

    output_csv = args.output_csv or f"results/qwen_{args.mode}_chunked_wer.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV: {output_csv}")

    if args.save_predictions:
        os.makedirs(os.path.dirname(args.save_predictions), exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            for row in prediction_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions: {args.save_predictions}")


if __name__ == "__main__":
    main()
