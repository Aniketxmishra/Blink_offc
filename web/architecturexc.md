# PHASE 2 — GPU Profiler & Benchmarking Engine

The GPU benchmarking engine acts as the primary data acquisition layer for Blink. It instruments hardware to capture truth data (execution times, memory footprints), pairs these labels with extracted model architecture features, and generates datasets to train the ML pipeline.

---

## 1. Profiling Scripts & Infrastructure
Blink's benchmarking pipeline resides primarily in two core modules:

* `blink/model_analyser.py` (`ModelAnalyzer`): Determines the **static** properties of the architecture without necessarily needing hardware constraints. It maps PyTorch layers, estimates naive FLOPs via `thop`, extracts bounding limits on Activation Memory, and decodes scaling KV cache sizes based on parameters footprint and batch depth.
* `scripts/collect_data.py` (`ModelProfiler`): The **dynamic** execution container that physically loads models into GPU VRAM, executes traces, averages execution steps, handles CUDA timeouts (OOMs), and outputs hardware telemetry to a labeled `.csv`. 
* *(Bonus)* `notebooks/lightning_llm_profiler_bonus20.py`: Extended profiler scripts utilized in data engineering capable of isolating deeper LLM component fingerprints.

---

## 2. Collected Metrics

Blink tracks runtime constraints across different sequence parameters (Prefill/Decode bounds for transformers, global pass for Vision nets). The following metrics are collected per hardware run:

### Time & Latency
- `execution_time_ms`: Average ms of a generic forward pass (For CNN / generic Transformers).
- `prefill_time_ms` (Time-To-First-Token latency): Captured by passing the full sequence context array natively through `model(use_cache=True)` simulating prefix injection.
- `decode_time_ms` (Time-Per-Output-Token): Captured by appending `1` sequence token onto the extending sequence maintaining an anchored KV cache object in memory. 
- `decode_tps`: Throughput metric calculated via `(batch_size * 1000) / max(decode_time, 0.001)`.

### Memory Footprint
- `peak_memory_mb`: Absolute highest memory watermark consumed by the GPU during the inference frame, extracted physically from the `torch.cuda.max_memory_allocated()` tensor. 
- `kv_cache_size_mb`: Memory constraint strictly computed from the keys and values memory space: `2 × batch × seq_len × seq_kv_heads × head_dim × dtype_size × num_layers`.

### Operations (Static)
- `flops`: Multi-accumulate (MACs) metric multiplied by 2 dynamically. Sceduled either through precise tracing natively (`thop.profile`) or approximated using basic hidden size dimension geometries.

---

## 3. Data Collection Pipeline Workflow

1. **Triggering the benchmark**: The user executes `scripts/collect_data.py --model-type all --batch-sizes 1 2 4 8 --seq-lengths 128 256` determining scope boundaries.
2. **Setup Layer**: The script preloads models natively using `torchvision.models` or `transformers.AutoModelForCausalLM`. It sweeps quantizations contexts (`fp32`, `fp16`, bitsandbytes `int8`, `int4`) via dynamic kwargs.
3. **Execution Loop**:
    - **Warmup Phase**: GPU cache paths are flushed actively via iterating `NUM_WARMUP_RUNS=3` non-recorded inferences. 
    - **Timing Phase**: Evaluator times exactly `NUM_TIMING_RUNS=5` loops enforcing strict sequence alignment via `torch.cuda.synchronize()`.
4. **Data Aggregation via Enrichment (Self-Contained DB Row)**: Results traverse back via `_enrich_llm_row()`. Missing properties are injected locally utilizing the generic `pynvml` OS driver:
    - GPU constants (`sm_count`, `memory_bandwidth_gbps`) are tagged against the execution run row ensuring the dataset correlates physical constraints vs bounds.
5. **Output Schema**: The pipeline produces a flat pandas `.csv` containing combinations of `[Hardware Specs] + [Model Meta Architecture Features] + [Static Compute Estimations] + [Physically Collected Validation Metrics]`. 

---

## 4. Model Architecture Variants (14+ Families)

The dataset leverages disparate architectures enforcing broad generalization bounds for its XGBoost engine.

**Vision / CNNs:**  
`ResNet18`, `ResNet50`, `VGG16`, `MobileNetV2`, `DenseNet`, `EfficientNet`, `MaxViT`, Swift / Base ViTs (`vit_b_16`, `vit_l`, `swin_t`).

**Transformers (Encoders):**  
`BERT-Base`, `RoBERTa-Base`

**Causal LLMs (Decoders):**  
`LLaMA-2`, `LLaMA-3`, `Mistral-7B`, `Gemma-2B/7B`, `Qwen1.5`, `GPT-2`, `TinyLlama`, `Pythia` (multiple scales), `OPT`, `Phi-2`, `Falcon-1B`.

### Architectural Fingerprint Constraints
LLM architectural boundaries diverge severely, deeply impacting hardware constraints. The benchmarking suite isolates these through advanced config extraction tools tracking:
* **Attention Mechanism (`num_kv_heads` vs `num_attention_heads`)**: Detects `Multi-Head Attention (MHA)` (1:1 ratio), `Grouped-Query Attention (GQA)` (ratio grouping), and `Multi-Query Attention (MQA)` (Falcon / Gemma implementations utilizing strict singular KV caching bounds).
* **Positional Embeddings (`pe_type`)**: Maps implementations leveraging Rotary Position Embeddings (`RoPE`) vs Attention with Linear Biases (`ALiBi`) or generic `Absolute` paths. 
* **Feed Forward Networks (`ffn_type`)**: Identifies standard `GELU` vs computationally bounded variant arrays utilizing `SwiGLU` projection channels.
* **Normalization Methods**: Identifying constraints mapping `RMSNorm` structures vs standard `LayerNorm`.

---

## 5. Low-Level Hooks & CUDA Instrumentation

Blink operates atop native PyTorch hardware APIs rather than deeply embedded native C++ / NvSci integrations. The pipeline prevents metric pollution by invoking standard driver hooks across execution layers:

* `torch.cuda.empty_cache()`: Erases GPU memory remnants between tests protecting metrics isolation.
* `torch.cuda.reset_peak_memory_stats(device)`: Acts as an origin marker validating the memory trace before forward execution kicks off.
* `torch.cuda.max_memory_allocated(device)`: Directly extracts the physical memory footprint of allocations across the last recorded span limit solving allocation boundaries natively without requiring manual Tensor `sys.getsizeof()` logic.
* `torch.cuda.synchronize()`: Crucially binds host CPU Python execution clocks explicitly with the delayed Asynchronous stream execution inside internal GPU pipelines. Without this, operations like `time.perf_counter()` would only index queueing speeds, not computational lengths. 
* Hardware metadata metrics (`gpu_name`, `tflops_fp32`) are read via `pynvml.nvmlDeviceGetHandleByIndex`.

_End of Phase 2._
