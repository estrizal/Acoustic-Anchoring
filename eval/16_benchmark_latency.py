"""
Benchmark Latency and Hardware Stats for Edge Deployment

This script evaluates the real-time factor (RTF) and token generation latency 
of the Acoustic Anchoring pipeline on the current hardware (CPU or GPU).
It logs exact CPU/RAM architecture to prove edge viability.
"""

import os
import platform
import time
import psutil

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Local imports
import sys
sys.path.insert(0, os.path.abspath("."))
from chunked_char_ipa_frontend import load_frontend, words_to_boundary_prompt

CTC_WEIGHTS = "models/phonemic_char_ipa_FULL_best.pt"
VOCAB_PATH = "data/char_ipa_vocab.json"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_CHECKPOINT = "models/lora_word_boundary/checkpoint-12612"

SYSTEM_PROMPT = (
    "You are an expert phonetic decoder. Convert the following word-bounded "
    "IPA string back into standard English."
)


def get_hardware_stats():
    print("=" * 60)
    print(" HARDWARE & ENVIRONMENT STATS")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    
    if cpuinfo:
        info = cpuinfo.get_cpu_info()
        print(f"CPU: {info.get('brand_raw', 'Unknown')}")
        print(f"Arch: {info.get('arch_string_raw', 'Unknown')}")
    else:
        print(f"CPU: {platform.processor()}")
    
    print(f"Logical Cores: {psutil.cpu_count(logical=True)}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    
    ram = psutil.virtual_memory()
    print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
    print(f"Available RAM: {ram.available / (1024**3):.2f} GB")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    return device


def benchmark_pipeline(device):
    print("Loading Acoustic Bridge...")
    t0 = time.time()
    ctc_model, ipa_tokenizer = load_frontend(CTC_WEIGHTS, VOCAB_PATH, device)
    print(f"Loaded Acoustic Bridge in {time.time() - t0:.2f}s")
    
    print("Loading Qwen-0.5B LLM + LoRA...")
    t0 = time.time()
    if device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, torch_dtype=torch.float16, device_map="cuda:0")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL)
    
    llm = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
    llm.eval()
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    print(f"Loaded LLM in {time.time() - t0:.2f}s\n")
    
    # Generate 5 seconds of dummy audio (16kHz)
    duration_s = 5.0
    sample_rate = 16000
    print(f"Benchmarking with {duration_s}s of dummy audio...")
    dummy_audio = torch.randn(1, int(sample_rate * duration_s)).to(device)
    
    # 1. Acoustic Frontend Latency
    print("\n--- Phase 1: Acoustic Bridge (Audio -> Phonemes) ---")
    t0 = time.time()
    import whisper as openai_whisper
    with torch.no_grad():
        padded = openai_whisper.pad_or_trim(dummy_audio.squeeze(0))
        mel = openai_whisper.log_mel_spectrogram(padded).unsqueeze(0).to(device)
        logits = ctc_model(mel)
        pred_ids = logits.argmax(dim=-1)[0].cpu().numpy()
        
    acoustic_latency = time.time() - t0
    acoustic_rtf = acoustic_latency / duration_s
    print(f"Acoustic Latency: {acoustic_latency:.3f}s")
    print(f"Acoustic RTF:     {acoustic_rtf:.3f}x real-time")
    
    # Simulate a 10-word decoded phoneme sequence
    dummy_words = [("hɛloʊ", 0.99), ("wɝld", 0.95), ("ðɪs", 0.98), ("ɪz", 0.99), 
                   ("ə", 0.90), ("tɛst", 0.92), ("ʌv", 0.91), ("ði", 0.98), 
                   ("ɛdʒ", 0.85), ("pɑɪplɑɪn", 0.95)]
    prompt = words_to_boundary_prompt(dummy_words)
    chat_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    # 2. LLM Latency
    print("\n--- Phase 2: LLM Corrector (Phonemes -> Text) ---")
    t0 = time.time()
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    llm_latency = time.time() - t0
    in_len = inputs.input_ids.size(1)
    new_tokens = out.size(1) - in_len
    tokens_per_sec = new_tokens / llm_latency
    
    print(f"LLM Latency:      {llm_latency:.3f}s")
    print(f"Tokens generated: {new_tokens}")
    print(f"Generation speed: {tokens_per_sec:.1f} tokens/sec")
    
    total_latency = acoustic_latency + llm_latency
    print("\n" + "=" * 60)
    print(f" TOTAL LATENCY FOR {duration_s}s AUDIO: {total_latency:.3f}s")
    print(f" TOTAL SYSTEM RTF: {total_latency / duration_s:.3f}x")
    print("=" * 60)
    
    # 3. Whisper Tiny Baseline
    print("\n--- Phase 3: Whisper Tiny Baseline (For Comparison) ---")
    print("Loading Whisper Tiny...")
    t0 = time.time()
    whisper_model = openai_whisper.load_model("tiny", device=device)
    print(f"Loaded Whisper Tiny in {time.time() - t0:.2f}s")
    
    t0 = time.time()
    with torch.no_grad():
        _ = whisper_model.transcribe(dummy_audio.squeeze(0).cpu().numpy() if device == "cuda" else dummy_audio.squeeze(0).numpy())
    whisper_latency = time.time() - t0
    whisper_rtf = whisper_latency / duration_s
    
    print(f"Whisper Tiny Latency: {whisper_latency:.3f}s")
    print(f"Whisper Tiny RTF:     {whisper_rtf:.3f}x real-time")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    device = get_hardware_stats()
    benchmark_pipeline(device)
