"""
Script: 05_verify_qwen_no_masking_wer.py
Paper claim: "No prompt masking (ckpt-500) → 31.41% WER [30.76, 32.15] on test-clean (N=2,620)"

This evaluates checkpoint-500 from training WITHOUT completion-only prompt masking.
The model was trained to predict ALL tokens (including the system+IPA prefix),
causing it to learn to echo the prompt instead of transcribing.

Run from project root:
    python reproducible/05_verify_qwen_no_masking_wer.py

Files needed:
    models/phonemic_ctc_best.pt           — ConvCTCAdapter weights
    data/ipa_vocab.json                   — IPA vocabulary
    lora_no_confidence_models/checkpoint-500/
        OR any equivalent "no-masking" checkpoint you saved.
    data/LibriSpeech/test-clean/          — Downloaded automatically if absent

NOTE: checkpoint-500 (no masking) may not be in the lora_no_confidence_models folder.
      Check the archive/ folder or the original training run output.
      If missing, this script exits with a clear error message.
"""

import os, sys, glob, torch, numpy as np, soundfile as sf, jiwer
import torch.nn as nn
import whisper as openai_whisper
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

# ── File paths ─────────────────────────────────────────────────────────────────
CTC_WEIGHTS      = "models/phonemic_ctc_best.pt"
VOCAB_PATH       = "data/ipa_vocab.json"
DATA_ROOT        = "./data"
LIBRI_ROOT       = "./data/LibriSpeech/test-clean"
QWEN_MODEL       = "Qwen/Qwen2.5-0.5B-Instruct"
# Update this path if the no-masking checkpoint is stored elsewhere:
NO_MASK_CKPT     = "outputs/checkpoint-500"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
EXPECTED_WER = 31.41
EXPECTED_CI  = (30.76, 32.15)
TOLERANCE    = 1.5
SYSTEM_PROMPT = "You are an expert phonetic decoder. Convert the following IPA string back into standard English."

# ── Shared model classes ───────────────────────────────────────────────────────
class IPAVocab:
    def __init__(self, path):
        import json
        with open(path, encoding="utf-8") as f: d = json.load(f)
        self.phone2id = d['phone2id']
        self.id2phone = {int(v): k for k, v in self.phone2id.items()}
        self.blank_id = 0; self.space_id = 1; self.vocab_size = len(self.phone2id)

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

def greedy_ctc(logits, vocab):
    import re
    ids=[int(i) for i in torch.softmax(logits,dim=-1)[0].argmax(dim=-1).tolist()]
    out,prev=[],None
    for i in ids:
        if i==vocab.blank_id: prev=i; continue
        if i==prev: continue
        out.append(' ' if i==vocab.space_id else vocab.id2phone.get(i,''))
        prev=i
    return re.sub(r' +',' ',''.join(out)).strip()

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

def bootstrap_ci(errors, lengths, n_boot=1000, seed=42):
    np.random.seed(seed); n=len(errors); wers=[]
    for _ in range(n_boot):
        idx=np.random.choice(n,n,replace=True)
        e,l=np.sum(errors[idx]),np.sum(lengths[idx])
        wers.append(e/l if l>0 else 0)
    wers.sort(); return wers[25]*100, wers[975]*100

def norm(text):
    import re
    return re.sub(r'\s+',' ',text.lower().translate(str.maketrans('','','.,?!;:"\''))).strip()

def main():
    print("="*60)
    print("CLAIM 05 — No Prompt Masking WER (ckpt-500)")
    print(f"  Expected: {EXPECTED_WER}% [{EXPECTED_CI[0]}, {EXPECTED_CI[1]}]  (N=2,620)")
    print("="*60)

    if not os.path.exists(NO_MASK_CKPT):
        print(f"\n  [ERROR] Checkpoint not found: {NO_MASK_CKPT}")
        print("  This checkpoint was produced by a training run WITHOUT completion-only masking.")
        print("  Check archive/ or your training logs. Alternatively, re-train with")
        print("  masking=False and save checkpoint at step 500.")
        sys.exit(1)

    for path in [CTC_WEIGHTS, VOCAB_PATH]:
        if not os.path.exists(path): print(f"  [ERROR] Missing: {path}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    import torchaudio
    torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    pairs = load_pairs(LIBRI_ROOT)
    print(f"  Loaded {len(pairs)} test-clean pairs.")

    vocab = IPAVocab(VOCAB_PATH)
    ctc_model = WhisperCTC(vocab.vocab_size).to(device)
    ctc_model.ctc_head.load_state_dict(torch.load(CTC_WEIGHTS,map_location=device,weights_only=False))
    ctc_model.eval()

    ipa_dataset = []
    for audio_path, ref in tqdm(pairs, desc="CTC decoding"):
        audio,sr=sf.read(audio_path,dtype="float32")
        if audio.ndim>1: audio=audio.mean(axis=1)
        tls=len(audio)/16000.0
        audio=openai_whisper.pad_or_trim(audio.astype(np.float32))
        mel=openai_whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
        with torch.no_grad(): logits=ctc_model(mel)
        ipa_dataset.append((ref, greedy_ctc(logits[:,:int(tls*50),:], vocab)))

    del ctc_model; torch.cuda.empty_cache()

    print(f"\n  Loading base Qwen + no-masking LoRA from: {NO_MASK_CKPT}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, dtype=torch.float16, device_map="cuda:0")
    llm = PeftModel.from_pretrained(base, NO_MASK_CKPT)
    llm.eval()

    errors, lengths = [], []
    bs = 32
    all_refs=[r for r,_ in ipa_dataset]; all_ipas=[i for _,i in ipa_dataset]
    for start in tqdm(range(0, len(ipa_dataset), bs), desc="No-mask Qwen WER"):
        b_ipas=all_ipas[start:start+bs]; b_refs=all_refs[start:start+bs]
        prompts=[f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{ipa}<|im_end|>\n<|im_start|>assistant\n" for ipa in b_ipas]
        inputs=tokenizer(prompts,return_tensors="pt",padding=True).to(llm.device)
        with torch.no_grad(): out=llm.generate(**inputs,max_new_tokens=256,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        for j,ref in enumerate(b_refs):
            in_len=inputs.input_ids[j].size(0)
            hyp=norm(tokenizer.decode(out[j][in_len:],skip_special_tokens=True))
            res=jiwer.process_words(norm(ref),hyp)
            errors.append(res.substitutions+res.deletions+res.insertions)
            lengths.append(len(norm(ref).split()))

    errors,lengths=np.array(errors),np.array(lengths)
    wer_val=np.sum(errors)/np.sum(lengths)*100
    ci_lo,ci_hi=bootstrap_ci(errors,lengths)

    print(f"\n  Measured WER: {wer_val:.2f}%  (95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%])")
    print(f"  Paper claims: {EXPECTED_WER}%  [{EXPECTED_CI[0]}, {EXPECTED_CI[1]}]")
    status=PASS if abs(wer_val-EXPECTED_WER)<=TOLERANCE else FAIL
    print(f"  Result: {status}\n")

if __name__ == "__main__":
    main()
