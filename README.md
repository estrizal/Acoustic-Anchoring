# Acoustic Anchoring

Official repository for **Acoustic Anchoring**, accepted at **IEEE SLT 2026 (Demo Track)**.

![Pipeline Overview](assets/pipeline_figure.png)

Acoustic Anchoring is a decoupled automatic speech recognition architecture for edge devices that already host a resident large language model (LLM). Instead of loading a second full-scale ASR model, our system attaches a compact acoustic bridge (39 MB) to the resident LLM, which performs phoneme-to-grapheme transcription via a plain-text forward pass. 

Under a strict 50 MB additive memory budget (e.g., smart glasses running an LLM), Acoustic Anchoring is the only deployable system in our benchmarks.

## Results (LibriSpeech test-clean)

| System | WER | Additive Memory Cost |
| :--- | :--- | :--- |
| Whisper Tiny | 5.60% | 75 MB |
| Whisper Base | 4.20% | 145 MB |
| Wav2Vec2-base | 3.70% | 360 MB |
| **Acoustic Anchoring (SPP-boundary)** | **13.43%** | **39 MB** |
| GPT-4o zero-shot oracle | 8.92% | N/A |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/acoustic-anchoring.git
cd acoustic-anchoring

# For Edge / CPU-only laptops (prevents downloading 3GB+ of NVIDIA CUDA binaries):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# For machines with NVIDIA GPUs:
# pip install -r requirements.txt
```

## Download Checkpoints

To run the system offline, you must download the pre-trained Acoustic Bridge and the fine-tuned LLMs.

1. Download the weights from [HuggingFace Hub](https://huggingface.co/YOUR_USERNAME/acoustic-anchoring) (Update link when uploaded).
2. Place the weights inside the `models/` directory:
   - `models/phonemic_char_ipa_FULL_best.pt` (39 MB) — The Acoustic Bridge
   - `models/lora_word_boundary/checkpoint-12612/` (33 MB) — SPP-boundary LoRA Adapter (13.43% WER)
   - `models/lora_yes_confidence_models/checkpoint-12612/` (33 MB) — SPP-confidence (13.49% WER)
   - `models/lora_placebo_confidence_models/checkpoint-12612/` (33 MB) — SPP-placebo (13.46% WER)

## Running Evaluations

We provide exact scripts to reproduce the variants in Table 1 from the paper.

```bash
# 1. Evaluate the standalone acoustic frontend (ConvCTCAdapter) -> 25.28% WER
python eval/03_verify_convctc_wer.py

# 2. Evaluate SPP-boundary (Best format) -> 13.43% WER
python eval/15_verify_qwen_chunked_wer.py

# 3. Evaluate SPP-confidence -> 13.49% WER
python eval/09_verify_qwen_confidence_epochs_wer.py

# 4. Evaluate SPP-placebo -> 13.46% WER
python eval/11_verify_qwen_placebo_confidence_epochs_wer.py

# 5. Evaluate GPT-4o Diagnostic Ceiling -> 8.92% WER
# (Requires OPENAI_API_KEY environment variable)
python eval/07_verify_gpt4o_oracle_wer.py

# 6. Benchmark Edge Hardware Latency (RTF & Token Speed)
# Logs your CPU/RAM specs and tests inference latency for a 5-second audio chunk.
python eval/16_benchmark_latency.py
```

## Running the Live Demo

*(Demo script coming soon prior to the conference)*

To run the 3-station live demo locally on your machine:
```bash
python demo.py --audio sample.wav
```

## Citation
```bibtex
@inproceedings{Aditya2026AcousticAnchoring,
  title={Acoustic Anchoring},
  author={Aditya},
  booktitle={2026 IEEE Spoken Language Technology Workshop (SLT)},
  year={2026}
}
```
