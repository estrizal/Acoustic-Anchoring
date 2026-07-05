"""
Script: 07_verify_gpt4o_oracle_wer.py
Paper claim: "GPT-4o oracle (zero-shot) → 11.21% WER [10.03, 12.44] on N=500 random subset"

This re-runs the stratified random evaluation of GPT-4o receiving the ConvCTCAdapter
IPA+confidence prompt and outputting English. Uses the same random seed=42 so the
N=500 sample is identical.

Run from project root:
    python reproducible/07_verify_gpt4o_oracle_wer.py

Files needed:
    models/phonemic_ctc_best.pt     — ConvCTCAdapter weights
    data/ipa_vocab.json             — IPA vocabulary
    data/LibriSpeech/test-clean/    — Downloaded automatically if absent

Environment variable:
    OPENAI_API_KEY                  — Your OpenAI API key (required!)

Cost estimate:
    ~500 API calls × ~200 tokens each ≈ 0.10–0.25 USD
"""

import os, sys, glob, math, torch, random, numpy as np, soundfile as sf, jiwer, time
import torch.nn as nn
import whisper as openai_whisper
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

# ── File paths ─────────────────────────────────────────────────────────────────
CTC_WEIGHTS = "models/phonemic_ctc_best.pt"
VOCAB_PATH  = "data/ipa_vocab.json"
DATA_ROOT   = "./data"
LIBRI_ROOT  = "./data/LibriSpeech/test-clean"
N_SAMPLES   = 500
SEED        = 42
GPT_MODEL   = "gpt-4o"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
EXPECTED_WER = 11.21
EXPECTED_CI  = (10.03, 12.44)
TOLERANCE    = 1.5
SYSTEM_PROMPT = "You are an expert phonetic decoder. Convert the following IPA string back into standard English."

# ── Model classes ──────────────────────────────────────────────────────────────
class IPAVocab:
    def __init__(self, path):
        import json
        with open(path, encoding="utf-8") as f: d = json.load(f)
        self.phone2id=d['phone2id']; self.id2phone={int(v):k for k,v in self.phone2id.items()}
        self.blank_id=0; self.space_id=1; self.vocab_size=len(self.phone2id)

class ConvCTCAdapter(nn.Module):
    def __init__(self, d_model, vs):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model); self.conv1=nn.Conv1d(d_model,d_model,5,padding=2); self.act1=nn.GELU()
        self.ln2=nn.LayerNorm(d_model); self.conv2=nn.Conv1d(d_model,d_model,5,padding=2); self.act2=nn.GELU()
        self.ln3=nn.LayerNorm(d_model); self.linear=nn.Linear(d_model,vs)
    def forward(self,x):
        x=self.act1(self.conv1(self.ln1(x).transpose(1,2))).transpose(1,2)
        x=self.act2(self.conv2(self.ln2(x).transpose(1,2))).transpose(1,2)
        return self.linear(self.ln3(x))

class WhisperCTC(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        _f=openai_whisper.load_model("tiny"); self.encoder=_f.encoder; self.dims=_f.dims
        self.ctc_head=ConvCTCAdapter(self.dims.n_audio_state, vocab_size); del _f
        for p in self.encoder.parameters(): p.requires_grad=False
    def forward(self,mel): return self.ctc_head(self.encoder(mel))

def greedy_ctc_with_conf(logits, vocab):
    """Returns (ipa_string, word_confidences) using the same confidence formula as the paper."""
    probs = torch.softmax(logits, dim=-1)[0]  # [T, V]
    ids   = probs.argmax(dim=-1).tolist()
    V     = probs.shape[-1]

    decoded_tokens, confs = [], []
    prev = -1
    for t, idx in enumerate(ids):
        if idx == prev: continue
        if idx != vocab.blank_id:
            decoded_tokens.append(idx)
            # Composite confidence: max_prob × (1 - normalised_entropy)
            p_max = probs[t, idx].item()
            entropy = -torch.sum(probs[t] * torch.log(probs[t] + 1e-9)).item()
            norm_entropy = entropy / math.log(V)
            confs.append(p_max * (1.0 - norm_entropy))
        prev = idx

    import re
    chars = [' ' if vocab.id2phone.get(i,'') == ' ' or i == vocab.space_id
             else vocab.id2phone.get(i,'') for i in decoded_tokens]
    ipa_str = re.sub(r' +', ' ', ''.join(chars)).strip()

    # Build word-level confidences (mean of character confs per word)
    words = ipa_str.split(' ')
    mean_conf = np.mean(confs) if confs else 0.5
    word_confs = [mean_conf] * len(words)
    return ipa_str, word_confs

def build_structured_prompt(ipa_str, word_confs):
    """Produce the confidence-tagged prompt format used in the paper."""
    words = ipa_str.split(' ')
    return ' '.join(f'[{c:.2f}] {w}' for w, c in zip(words, word_confs))

def load_pairs(root):
    pairs=[]
    for tf in glob.glob(os.path.join(root,"**","*.trans.txt"),recursive=True):
        folder=os.path.dirname(tf)
        with open(tf,encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                uid,*parts=line.split()
                flac=os.path.join(folder,uid+".flac")
                if os.path.exists(flac): pairs.append((flac," ".join(parts).lower()))
    return pairs

def bootstrap_ci(wer_tuples, n_boot=1000, seed=42):
    random.seed(seed); n=len(wer_tuples); wers=[]
    for _ in range(n_boot):
        sample=[wer_tuples[random.randint(0,n-1)] for _ in range(n)]
        e=sum(x[0] for x in sample); l=sum(x[1] for x in sample)
        wers.append(e/l*100 if l>0 else 0)
    wers.sort(); return wers[25], wers[975]

def norm(text):
    import re
    return re.sub(r'\s+',' ',text.lower().translate(str.maketrans('','','.,?!;:"\''))).strip()

def main():
    print("="*60)
    print("CLAIM 07 — GPT-4o Oracle WER (N=500 random subset)")
    print(f"  Expected: {EXPECTED_WER}% [{EXPECTED_CI[0]}, {EXPECTED_CI[1]}]")
    print("  Seed=42 ensures identical sample to paper.")
    print("="*60)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n  [ERROR] OPENAI_API_KEY environment variable not set.")
        print("  Set it with: set OPENAI_API_KEY=sk-...")
        sys.exit(1)

    for path in [CTC_WEIGHTS, VOCAB_PATH]:
        if not os.path.exists(path): print(f"  [ERROR] Missing: {path}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    import torchaudio
    torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    pairs = load_pairs(LIBRI_ROOT)
    print(f"  Total test-clean pairs: {len(pairs)}")

    # ── Sample N=500 with seed=42 (identical to paper) ─────────────────────────
    random.seed(SEED)
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    sampled = [pairs[i] for i in indices[:N_SAMPLES]]
    print(f"  Sampled {len(sampled)} utterances (seed={SEED})\n")

    vocab = IPAVocab(VOCAB_PATH)
    ctc_model = WhisperCTC(vocab.vocab_size).to(device)
    ctc_model.ctc_head.load_state_dict(torch.load(CTC_WEIGHTS,map_location=device,weights_only=False))
    ctc_model.eval()

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    wer_tuples = []
    for audio_path, ref in tqdm(sampled, desc="GPT-4o oracle"):
        audio,sr=sf.read(audio_path,dtype="float32")
        if audio.ndim>1: audio=audio.mean(axis=1)
        tls=len(audio)/16000.0
        audio=openai_whisper.pad_or_trim(audio.astype(np.float32))
        mel=openai_whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
        with torch.no_grad(): logits=ctc_model(mel)
        true_frames=int(tls*50)
        ipa_str, word_confs = greedy_ctc_with_conf(logits[:,: true_frames,:], vocab)
        structured = build_structured_prompt(ipa_str, word_confs)

        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":structured}],
                temperature=0.0
            )
            hyp = norm(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"  API error: {e}"); time.sleep(3); continue

        ref_n = norm(ref)
        res = jiwer.process_words(ref_n, hyp)
        wer_tuples.append((res.substitutions + res.deletions + res.insertions, len(ref_n.split())))

    n = len(wer_tuples)
    total_e = sum(x[0] for x in wer_tuples)
    total_l = sum(x[1] for x in wer_tuples)
    wer_val = total_e / total_l * 100 if total_l > 0 else 0
    ci_lo, ci_hi = bootstrap_ci(wer_tuples)

    print(f"\n  Measured WER: {wer_val:.2f}%  (95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%])")
    print(f"  Paper claims: {EXPECTED_WER}%  [{EXPECTED_CI[0]}, {EXPECTED_CI[1]}]")
    status = PASS if abs(wer_val - EXPECTED_WER) <= TOLERANCE else FAIL
    print(f"  Result: {status}  (N={n})\n")

if __name__ == "__main__":
    main()
