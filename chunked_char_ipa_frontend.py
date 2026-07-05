"""
Chunked char-IPA frontend for Whisper-Tiny CTC inference.

Whisper's encoder accepts 30 s windows. The old evaluation scripts padded or
trimmed every utterance into one window, which truncates the small test-clean
tail above 30 s. This module keeps the existing single-window behavior for
short clips and uses overlapping windows for longer clips.
"""

import json
import os
import re

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import whisper as openai_whisper


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


def load_frontend(weights_path, vocab_path, device):
    tokenizer = CharIPATokenizer()
    tokenizer.load(vocab_path)
    model = WhisperForPhonemicCTC(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    return model, tokenizer


def read_mono_16k(path):
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr}: {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def ctc_words_with_confidence(logits, tokenizer):
    probs = torch.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1).tolist()
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    log_v = np.log(tokenizer.vocab_size)
    composite = probs.max(dim=-1).values * (1.0 - entropy / log_v)

    words = []
    current_chars = []
    current_confs = []
    prev = None
    space_id = tokenizer.char2id.get("<SPACE>")

    for frame_idx, tid in enumerate(pred_ids):
        if tid == tokenizer.blank_id:
            prev = tid
            continue
        if tid == prev:
            continue
        if tid == space_id:
            if current_chars:
                words.append(("".join(current_chars), float(np.mean(current_confs))))
                current_chars = []
                current_confs = []
        else:
            char = tokenizer.id2char.get(tid, "")
            if char:
                current_chars.append(char)
                current_confs.append(float(composite[frame_idx]))
        prev = tid

    if current_chars:
        words.append(("".join(current_chars), float(np.mean(current_confs))))
    return words


def merge_word_chunks(chunks, max_overlap_words=24):
    merged = []
    for chunk_words in chunks:
        if not chunk_words:
            continue
        if not merged:
            merged.extend(chunk_words)
            continue

        merged_text = [word for word, _ in merged]
        chunk_text = [word for word, _ in chunk_words]
        max_k = min(max_overlap_words, len(merged_text), len(chunk_text))
        overlap = 0
        for k in range(max_k, 0, -1):
            if merged_text[-k:] == chunk_text[:k]:
                overlap = k
                break
        merged.extend(chunk_words[overlap:])
    return merged


def audio_windows(audio, sample_rate=16000, window_s=25.0, overlap_s=2.0):
    window = int(window_s * sample_rate)
    hop = int((window_s - overlap_s) * sample_rate)
    if len(audio) <= sample_rate * 30:
        yield audio, len(audio) / sample_rate, 0.0
        return
    start = 0
    is_first = True
    while start < len(audio):
        end = min(start + window, len(audio))
        trim_start_s = 0.0 if is_first else overlap_s
        yield audio[start:end], (end - start) / sample_rate, trim_start_s
        if end == len(audio):
            break
        start += hop
        is_first = False


def decode_audio_to_words(
    audio_path,
    ctc_model,
    tokenizer,
    device,
    chunked=True,
    window_s=25.0,
    overlap_s=2.0,
):
    audio = read_mono_16k(audio_path)
    if not chunked:
        chunk_audio = openai_whisper.pad_or_trim(audio)
        mel = openai_whisper.log_mel_spectrogram(chunk_audio).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = ctc_model(mel)[0, : int(len(audio) / 16000.0 * 50), :]
        return ctc_words_with_confidence(logits, tokenizer)

    chunks = []
    for chunk_audio, chunk_len_s, trim_start_s in audio_windows(audio, window_s=window_s, overlap_s=overlap_s):
        padded = openai_whisper.pad_or_trim(chunk_audio)
        mel = openai_whisper.log_mel_spectrogram(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            start_frame = int(trim_start_s * 50)
            end_frame = int(chunk_len_s * 50)
            logits = ctc_model(mel)[0, start_frame:end_frame, :]
        chunks.append(ctc_words_with_confidence(logits, tokenizer))
    return merge_word_chunks(chunks)


def words_to_plain_ipa(words):
    return re.sub(r" +", " ", " ".join(word for word, _ in words)).strip()


def words_to_confidence_prompt(words):
    return " ".join(f"[{conf:.2f}] {word}" for word, conf in words)


def words_to_placebo_prompt(words, value="0.50"):
    return " ".join(f"[{value}] {word}" for word, _ in words)


def words_to_boundary_prompt(words):
    return " ".join(f"<w> {word} </w>" for word, _ in words)
