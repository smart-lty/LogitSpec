# LogitSpec: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation

Official implementation of the paper *LogitSpec: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation* (arXiv: [2507.01449](https://arxiv.org/abs/2507.01449)). This repository ships the draft-then-verify decoding algorithm, inference harnesses, and speed benchmarking utilities used in the paper, tailored for the Llama 2 family (Llama 2, Vicuna, CodeLlama, …).

## Why LogitSpec?
- **Next-next-token speculation** &mdash; reuses n-gram caches from the prompt to propose multi-branch drafts without training auxiliary draft models.
- **Retrieval-aware draft tree** &mdash; dynamically grows candidate paths with configurable capacity, balancing acceptance length and compute.
- **Turn-key benchmarking** &mdash; FastChat-based evaluation loop plus Spec-Bench style speed profiler capture new-token counts, wall-clock latency, and accept-length histograms.

### Repository Tour
```
LogitSpec/
├── data/                # Spec-Bench, CNN/DM, GSM8K, HumanEval question sets
├── evaluation/          # Inference pipelines and speed analysis
├── model/logitspec/     # Pool builder, KV cache, and patched Llama kernels
├── eval.sh              # End-to-end reproduction script
├── requirements.txt     # Python dependencies
└── README.md
```

## Environment Setup
1. **Create a dedicated environment (Python 3.9 recommended):**
   ```bash
   conda create -n logitspec python=3.9
   conda activate logitspec
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > `torch` is intentionally left unpinned. Install the CUDA build that matches your driver and is compatible with `transformers==4.37.1`.
3. **Hardware guidance:**
   - Single 24 GB GPU suffices for 7B models; larger models or multi-run sweeps benefit from 48 GB+.
   - CUDA 11.8 or newer is recommended. Optional extras such as `flash-attn`/`xformers` can further reduce latency if your stack supports them.

## Models and Datasets
- **Model weights:** Download your target checkpoint (e.g., `lmsys/vicuna-7b-v1.3`) via `huggingface-cli download` or `git lfs`, and point `--model-path` (or `Vicuna_PATH` in `eval.sh`) to the local directory.
- **Benchmarks in-tree:**
  - `data/spec_bench/question.jsonl` (Spec-Bench full set)
  - `data/cnndm/question.jsonl`, `data/gsm8k/question.jsonl`, `data/humaneval/question.jsonl`, …
- **Custom benchmarks:** drop a `question.jsonl` into `data/<bench_name>/` and reference it via `--bench-name <bench_name>`.

## Quick Reproduction
`eval.sh` orchestrates the full workflow. After editing `Vicuna_PATH`:
```bash
sh eval.sh
```
This sequentially runs the baseline decoder, LogitSpec decoder, and the speed reporter, writing JSONL traces under `data/<bench_name>/model_answer/`.

### Manual Control (recommended for sweeps)
**1. Baseline decoding**
```bash
python -m evaluation.inference_baseline \
  --model-path /path/to/vicuna-7b-v1.3 \
  --model-id vicuna-7b-v1.3-vanilla-float16-temp-0.0 \
  --bench-name spec_bench \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --dtype float16
```
**2. LogitSpec decoding**
```bash
python -m evaluation.inference_logitspec \
  --model-path /path/to/vicuna-7b-v1.3 \
  --model-id vicuna-7b-v1.3-logitspec-float16-temp-0.0 \
  --bench-name spec_bench \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --max_ngram_size 3 \
  --num_pred_tokens 20 \
  --draft_tree_capacity 64 \
  --dtype float16
```
**3. Throughput & accept-length statistics**
```bash
python evaluation/speed.py \
  --file-path data/spec_bench/model_answer/vicuna-7b-v1.3-logitspec-float16-temp-0.0.jsonl \
  --base-path data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
  --tokenizer-path /path/to/vicuna-7b-v1.3
```
Add `--mean-report` when aggregating multiple runs.

## Configuration Cheat Sheet
| Argument | Description | Typical range |
| --- | --- | --- |
| `--max-new-tokens` | Maximum continuation length per turn | 512–2048 |
| `--temperature` | Sampling temperature (LogitSpec currently optimized for greedy) | 0–0.1 |
| `--max_ngram_size` | Longest n-gram cached in the pool | 2–4 |
| `--num_pred_tokens` | Max tokens retrieved per candidate branch | 8–32 |
| `--draft_tree_capacity` | Total tokens allowed in the draft tree per step | 32–128 |
| `--num-gpus-per-model` | GPUs allocated per process | 1 (default) |
| `--question-begin/end` | Evaluate a slice of the benchmark | custom |

> Tip: `draft_tree_capacity` and `num_pred_tokens` drive both acceptance length and memory. Smaller models benefit from conservative values to avoid OOM.

## Troubleshooting
1. **CUDA OOM** &mdash; Lower `--max-new-tokens`, `draft_tree_capacity`, or switch to `--dtype bfloat16`; if using `device_map="auto"`, pin the model to a specific GPU.
2. **Model not found** &mdash; Ensure weights are downloaded locally and the path is absolute. HF token authentication may be required for gated checkpoints.
3. **Evaluation stops early** &mdash; Remove `--question-begin`/`--question-end` overrides or double-check benchmark length.
4. **Speed script feels slow** &mdash; The current implementation re-tokenizes baseline outputs. For large sweeps, subsample the JSONL or adapt the script to cache encodings.

## Citation
If LogitSpec helps your research, please cite:
```bibtex
@misc{liu2025logitspecacceleratingretrievalbasedspeculative,
      title={LogitSpec: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation}, 
      author={Tianyu Liu and Qitan Lv and Hao Li and Xing Gao and Xiao Sun},
      year={2025},
      eprint={2507.01449},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.01449}, 
}
```

## Acknowledgements
This project builds upon Spec-Bench and FastChat. We are grateful to their authors and the broader open-source community for the foundations enabling LogitSpec.
