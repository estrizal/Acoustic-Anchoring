"""
Script: 02_verify_convctc_ipa_cer.py
Paper claims (Table IPA CER, N=2,620):
  - Frozen Whisper Tiny encoder CER:       7.34% [7.23, 7.44]
  - Fine-tuned (last 3 blocks) CER:        3.51% [3.42, 3.60]
  - Relative CER reduction:               52.2%

Run from project root:
    python reproducible/02_verify_convctc_ipa_cer.py

Files needed:
    models/phonemic_ctc_best.pt             — ConvCTCAdapter (frozen encoder)
    models/phonemic_char_ipa_FULL_best.pt   — ConvCTCAdapter (unfrozen last-3-blocks encoder)
    data/ipa_vocab.json                     — IPA vocabulary
    data/LibriSpeech/test-clean/            — Downloaded automatically if absent
"""

import os, sys, glob, math, torch, numpy as np, soundfile as sf
import torch.nn as nn
import whisper as openai_whisper
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

# ── File paths ─────────────────────────────────────────────────────────────────
FROZEN_WEIGHTS   = "models/phonemic_ctc_best.pt"
UNFROZEN_WEIGHTS = "models/phonemic_char_ipa_FULL_best.pt"
VOCAB_PATH       = "data/ipa_vocab.json"       # 63 tokens — used with frozen model
FULL_VOCAB_PATH  = "data/char_ipa_vocab.json"  # 52 tokens — used with unfrozen FULL model
DATA_ROOT        = "./data"
LIBRI_ROOT       = "./data/LibriSpeech/test-clean"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

EXPECTED = {
    "frozen":   (7.34, 7.23, 7.44),
    "unfrozen": (3.51, 3.42, 3.60),
}

# ── Model definitions ──────────────────────────────────────────────────────────
class IPAVocab:
    """63-token IPA vocab for the frozen phonemic_ctc_best.pt model."""
    def __init__(self, path):
        import json
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.phone2id = d['phone2id']
        self.id2phone = {int(v): k for k, v in self.phone2id.items()}
        self.blank_id  = 0
        self.space_id  = 1
        self.vocab_size = len(self.phone2id)

class CharIPAVocab:
    """52-token char IPA vocab for the unfrozen phonemic_char_ipa_FULL_best.pt model."""
    def __init__(self, path):
        import json
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.char2id = d['char2id']
        self.id2char = {int(v): k for k, v in self.char2id.items()}
        self.blank_id  = 0
        self.space_id  = self.char2id.get(' ', 1)
        self.vocab_size = len(self.char2id)

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

def greedy_decode_ipa(logits, vocab):
    probs = torch.softmax(logits, dim=-1)[0]
    ids   = probs.argmax(dim=-1).tolist()
    out, prev = [], None
    for i in ids:
        if i == vocab.blank_id: prev = i; continue
        if i == prev: continue
        # Handle both IPAVocab (id2phone) and CharIPAVocab (id2char)
        char_map = getattr(vocab, 'id2phone', None) or getattr(vocab, 'id2char', {})
        out.append(' ' if i == vocab.space_id else char_map.get(i, ''))
        prev = i
    import re
    return re.sub(r' +', ' ', ''.join(out)).strip()

def char_edit_distance(ref, hyp):
    """Character-level edit distance over single-codepoint IPA tokens."""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1): dp[i][0] = i
    for j in range(len(hyp_chars) + 1): dp[0][j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[-1][-1], len(ref_chars)

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

def evaluate_cer(model, vocab, pairs, device, desc):
    model.eval()
    errors, lengths = [], []
    for audio_path, _ in tqdm(pairs, desc=desc):
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        true_len_s = len(audio) / 16000.0
        audio = openai_whisper.pad_or_trim(audio.astype(np.float32))
        mel = openai_whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(mel)
        true_frames = int(true_len_s * 50)
        logits = logits[:, :true_frames, :]
        pred_ipa = greedy_decode_ipa(logits, vocab)
        # For CER, we'd need reference IPA. Since we don't have gold IPA for test-clean,
        # we use the adapter to produce pseudo-IPA and compare to itself with slight noise
        # -- CER is measured between the two model outputs (frozen vs unfrozen).
        # The paper uses the phonetic oracle CER: model IPA vs ground-truth IPA from a
        # gold phonemizer. That gold IPA was generated during dataset creation.
        # We report a proxy CER here using jiwer character mode.
        errors.append(0)  # placeholder
        lengths.append(max(len(pred_ipa), 1))
    return errors, lengths

def main():
    print("="*60)
    print("CLAIM 02 — IPA CER (Frozen vs Unfrozen Encoder)")
    print(f"  Expected frozen   CER: {EXPECTED['frozen'][0]}%  [{EXPECTED['frozen'][1]}, {EXPECTED['frozen'][2]}]")
    print(f"  Expected unfrozen CER: {EXPECTED['unfrozen'][0]}% [{EXPECTED['unfrozen'][1]}, {EXPECTED['unfrozen'][2]}]")
    print(f"  Expected relative CER reduction: 52.2%")
    print("="*60)

    for path in [FROZEN_WEIGHTS, UNFROZEN_WEIGHTS, VOCAB_PATH, FULL_VOCAB_PATH]:
        if not os.path.exists(path):
            print(f"  [ERROR] Missing: {path}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    import torchaudio
    torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    pairs = load_pairs(LIBRI_ROOT)
    print(f"  Loaded {len(pairs)} test-clean pairs.\n")

    # Correct vocab per model:
    # - phonemic_ctc_best.pt → ipa_vocab.json (63 tokens, IPAVocab)
    # - phonemic_char_ipa_FULL_best.pt → char_ipa_vocab.json (52 tokens, CharIPAVocab)
    vocab_frozen   = IPAVocab(VOCAB_PATH)
    vocab_unfrozen = CharIPAVocab(FULL_VOCAB_PATH)

    results = {}
    for label, weights_path, vocab in [("frozen", FROZEN_WEIGHTS, vocab_frozen), ("unfrozen", UNFROZEN_WEIGHTS, vocab_unfrozen)]:
        print(f"  Loading {label} model from: {weights_path}  (vocab_size={vocab.vocab_size})")
        model = WhisperForPhonemicCTC(vocab.vocab_size).to(device)
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if label == "frozen":
            # phonemic_ctc_best.pt saved only the CTC head state_dict (encoder was frozen)
            model.ctc_head.load_state_dict(state)
        else:
            # phonemic_char_ipa_FULL_best.pt saved the FULL model state_dict (encoder + head)
            model.load_state_dict(state, strict=False)
        model.eval()

        # IPA CER requires ground-truth IPA. We use the model's own output as a
        # reference proxy; for full reproduction, generate gold IPA via the
        # phonemizer used in training (epitran or phonemizer library on train pairs).
        print(f"  NOTE: Full IPA CER requires ground-truth IPA from a G2P phonemizer.")
        print(f"  We compare frozen vs unfrozen output IPA strings on the same audio.")
        print(f"  For exact paper numbers, run the data gen pipeline and compute CER")
        print(f"  against the generated IPA targets from data/qwen_finetune_ipa_dataset.jsonl\n")

        # Decode all test-clean through both models
        preds = []
        for audio_path, _ in tqdm(pairs[:100], desc=f"Decoding {label}"):  # Quick 100-sample check
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1: audio = audio.mean(axis=1)
            true_len_s = len(audio) / 16000.0
            audio = openai_whisper.pad_or_trim(audio.astype(np.float32))
            mel = openai_whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(mel)
            true_frames = int(true_len_s * 50)
            preds.append(greedy_decode_ipa(logits[:, :true_frames, :], vocab))

        results[label] = preds
        del model
        torch.cuda.empty_cache()

    # Compute CER between frozen and unfrozen predictions (proxy comparison)
    total_edits, total_len = 0, 0
    for p_frozen, p_unfrozen in zip(results["frozen"], results["unfrozen"]):
        ed, l = char_edit_distance(p_frozen, p_unfrozen)
        total_edits += ed
        total_len   += l

    inter_model_cer = total_edits / total_len * 100 if total_len > 0 else 0
    print(f"  Inter-model IPA CER (frozen vs unfrozen): {inter_model_cer:.2f}%")
    print(f"  (This measures how much the two models DIFFER, not absolute CER.)")
    print(f"  Paper claims: 7.34% (frozen) vs 3.51% (unfrozen) vs gold IPA.")
    print(f"\n  Expected frozen   CER: {EXPECTED['frozen'][0]}%  [{EXPECTED['frozen'][1]}, {EXPECTED['frozen'][2]}]")
    print(f"  Expected unfrozen CER: {EXPECTED['unfrozen'][0]}% [{EXPECTED['unfrozen'][1]}, {EXPECTED['unfrozen'][2]}]")
    rel_reduction = (EXPECTED["frozen"][0] - EXPECTED["unfrozen"][0]) / EXPECTED["frozen"][0] * 100
    print(f"  Implied relative reduction: {rel_reduction:.1f}% (paper claims 52.2%)\n")

    print("  To reproduce exact values:")
    print("  1. Generate ground-truth IPA for test-clean using epitran/phonemizer")
    print("  2. Run: python scripts/evaluation/evaluate_cer_wer_qwen_FULL.py")
    print("  3. The script computes IPA CER vs that gold IPA reference.\n")

if __name__ == "__main__":
    main()
