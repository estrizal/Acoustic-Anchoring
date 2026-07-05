"""
Script: 03_verify_convctc_wer.py
Paper claim: "ConvCTCAdapter (frozen enc.) → 25.28% WER [24.78, 25.79] on test-clean (N=2,620)"

This is the acoustic-only WER: just the Whisper encoder + ConvCTCAdapter,
decoded with greedy CTC, compared directly to ground-truth English text.
No Qwen LLM correction is applied.

Run from project root:
    python reproducible/03_verify_convctc_wer.py

Files needed:
    models/phonemic_ctc_best.pt     — ConvCTCAdapter weights (frozen encoder)
    data/ipa_vocab.json             — IPA vocabulary
    data/LibriSpeech/test-clean/    — Downloaded automatically if absent
"""

import os, sys, glob, math, torch, numpy as np, soundfile as sf, jiwer
import torch.nn as nn
import whisper as openai_whisper
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

# ── File paths ─────────────────────────────────────────────────────────────────
CTC_WEIGHTS = "models/phonemic_ctc_best.pt"
VOCAB_PATH  = "data/ipa_vocab.json"
DATA_ROOT   = "./data"
LIBRI_ROOT  = "./data/LibriSpeech/test-clean"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
EXPECTED_WER = 25.28
EXPECTED_CI  = (24.78, 25.79)
TOLERANCE    = 1.0

# ── Model definitions ──────────────────────────────────────────────────────────
class IPAVocab:
    def __init__(self, path):
        import json
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.phone2id = d['phone2id']
        self.id2phone = {int(v): k for k, v in self.phone2id.items()}
        self.blank_id  = 0
        self.space_id  = 1
        self.vocab_size = len(self.phone2id)

class ConvCTCAdapter(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln1    = nn.LayerNorm(d_model)
        self.conv1  = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.act1   = nn.GELU()
        self.ln2    = nn.LayerNorm(d_model)
        self.conv2  = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.act2   = nn.GELU()
        self.ln3    = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.ln1(x).transpose(1, 2)
        x = self.act1(self.conv1(x)).transpose(1, 2)
        x = self.ln2(x).transpose(1, 2)
        x = self.act2(self.conv2(x)).transpose(1, 2)
        x = self.ln3(x)
        return self.linear(x)

class WhisperForPhonemicCTC(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        _full = openai_whisper.load_model("tiny")
        self.encoder  = _full.encoder
        self.dims     = _full.dims
        self.ctc_head = ConvCTCAdapter(self.dims.n_audio_state, vocab_size)
        del _full
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, mel):
        return self.ctc_head(self.encoder(mel))

def greedy_ctc_decode(logits, vocab):
    """Greedy CTC decode: IPA token sequence → space-joined string."""
    probs = torch.softmax(logits, dim=-1)[0]
    ids   = probs.argmax(dim=-1).tolist()
    out, prev = [], None
    for i in ids:
        if i == vocab.blank_id: prev = i; continue
        if i == prev: continue
        out.append(' ' if i == vocab.space_id else vocab.id2phone.get(i, ''))
        prev = i
    import re
    return re.sub(r' +', ' ', ''.join(out)).strip()

def load_pairs(root):
    pairs = []
    for tf in glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True):
        folder = os.path.dirname(tf)
        with open(tf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                uid, *parts = line.split()
                flac = os.path.join(folder, uid + ".flac")
                if os.path.exists(flac):
                    pairs.append((flac, " ".join(parts).lower()))
    return pairs

def bootstrap_ci(errors, lengths, n_boot=1000, seed=42):
    np.random.seed(seed)
    n = len(errors)
    wers = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        e, l = np.sum(errors[idx]), np.sum(lengths[idx])
        wers.append(e / l if l > 0 else 0)
    wers.sort()
    return wers[25] * 100, wers[975] * 100

def normalize(text):
    import re
    return re.sub(r'\s+', ' ', text.lower().translate(str.maketrans('', '', '.,?!;:"\''))).strip()

def main():
    print("="*60)
    print("CLAIM 03 — ConvCTCAdapter WER (Acoustic-Only, Frozen Encoder)")
    print(f"  Expected: {EXPECTED_WER}% [{EXPECTED_CI[0]}, {EXPECTED_CI[1]}]  (N=2,620)")
    print("="*60)

    for path in [CTC_WEIGHTS, VOCAB_PATH]:
        if not os.path.exists(path):
            print(f"  [ERROR] Missing: {path}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    import torchaudio
    torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    pairs = load_pairs(LIBRI_ROOT)
    print(f"  Loaded {len(pairs)} test-clean pairs.\n")

    vocab = IPAVocab(VOCAB_PATH)
    model = WhisperForPhonemicCTC(vocab.vocab_size).to(device)
    state = torch.load(CTC_WEIGHTS, map_location=device, weights_only=False)
    model.ctc_head.load_state_dict(state)
    model.eval()
    print(f"  Loaded ConvCTCAdapter from: {CTC_WEIGHTS}\n")

    errors, lengths = [], []
    for audio_path, ref in tqdm(pairs, desc="ConvCTCAdapter WER"):
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        true_len_s = len(audio) / 16000.0
        audio = openai_whisper.pad_or_trim(audio.astype(np.float32))
        mel = openai_whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(mel)
        true_frames = int(true_len_s * 50)
        logits = logits[:, :true_frames, :]
        hyp = normalize(greedy_ctc_decode(logits, vocab))
        ref = normalize(ref)
        out = jiwer.process_words(ref, hyp)
        errors.append(out.substitutions + out.deletions + out.insertions)
        lengths.append(len(ref.split()))

    errors, lengths = np.array(errors), np.array(lengths)
    wer_val = np.sum(errors) / np.sum(lengths) * 100
    ci_lo, ci_hi = bootstrap_ci(errors, lengths)

    print(f"\n  Measured WER: {wer_val:.2f}%  (95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%])")
    print(f"  Paper claims: {EXPECTED_WER}%  [{EXPECTED_CI[0]}, {EXPECTED_CI[1]}]")
    status = PASS if abs(wer_val - EXPECTED_WER) <= TOLERANCE else FAIL
    print(f"  Result: {status}\n")

if __name__ == "__main__":
    main()
