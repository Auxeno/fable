<div align="center">

  <h1>📖 Fable</h1>
  
  <h3>A compact storytelling language model</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.12-636EFA.svg)](https://www.python.org/)
  [![JAX](https://img.shields.io/badge/JAX-0.8-AB63FA.svg)](https://github.com/google/jax)
  [![Flax NNX](https://img.shields.io/badge/Flax-NNX-00CCBB.svg)](https://github.com/google/flax/tree/main/flax/nnx)
  [![License](https://img.shields.io/badge/License-MIT-FECB52.svg)](licence)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auxeno/fable/blob/main/notebooks/fable-demo.ipynb)
  
</div>

---

<div align="center">
    <h3>
      <a href="#overview">Overview</a> |
      <a href="#demo">Demo</a> |
      <a href="#installation">Installation</a> |
      <a href="#text-generation">Text Generation</a> |
      <a href="#data-pipeline">Data Pipeline</a> |
      <a href="#training">Training</a>
    </h3>
</div>

---

## 🧭 Overview

Fable is a compact **storytelling language model** implemented from scratch using JAX and Flax NNX.  
It provides an end-to-end training pipeline — from text preparation and tokenisation to model training and autoregressive text generation.

Fable is designed to be **small and fast** — a fluent model can be trained on consumer GPUs in under an hour.  
The included demo checkpoint was trained for **~2 hours on an RTX 4090** using the default configuration.

---

## ✨ Features

- 🧠 **Minimal GPT Architecture** — Lightweight decoder-only transformer (~800 k parameters).  
- 🧵 **Simple Data Pipeline** — Deterministic download → clean → tokenise workflow for small text datasets.  
- 🚀 **JIT Compilation** — Core steps compiled with `jax.jit`, achieving ~3 million tokens/sec throughput.  
- 💾 **Orbax Checkpointing** — Save and restore model state and hyperparameters in a single file.  
- 📝 **Text Generation Tools** — Generate short stories with adjustable sampling temperature.  

---

## 🎬 Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auxeno/fable/blob/main/notebooks/fable-demo.ipynb)

The demo notebook walks through:
- Loading the pretrained demo checkpoint.  
- Generating short stories from prompts.  
- Exploring different temperature values.  
- Optionally running the data pipeline and a brief training step.  

The notebook runs entirely in a hosted Colab environment — no local setup required.

---

## 📦 Installation

To install locally:

```bash
# Install latest release
pip install git+https://github.com/auxeno/fable

# Local development setup
git clone https://github.com/auxeno/fable.git
cd fable
pip install -e .
```

For GPU acceleration, install the JAX wheel matching your CUDA version, for example:
```bash
pip install --upgrade "jax[cuda13]"
```

Refer to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for up-to-date instructions.

---

## 🧾 Text Generation

To generate a story, simply write the first few words/lines and Fable will continue.

### Python API

```python
from fable import generate_text

generate_text("Lily got a new puppy")  # Uses checkpoints/demo.ckpt by default
# To sample your own run: generate_text("...", checkpoint="model_state.ckpt")
```

### Command Line Interface

```bash
fable-generate --start "Lily got a new puppy" --temperature 0.6
# Use --checkpoint model_state.ckpt to load your own training run
```

### Temperature Examples

Sampling temperature controls the balance between determinism and creativity:

- **Low (≈0.4):** Predictable and faithful to training data.  
- **Medium (≈0.6):** More varied but occasionally incoherent.  
- **High (≈0.8):** Grammatically fluent but semantically unstable.

#### 🟦 Temperature 0.4 — Stable and consistent
```python
generate_text("Lily got a new puppy", temperature=0.4)

"""
Lily got a new puppy and said, "I want to see it!" Her mom smiled and said, 
"Okay, Lily. Let's go to the puppy and see if we can play with it." 
Lily smiled and said, "Okay."
Lily felt better.
...
"""
```

#### 🟧 Temperature 0.6 — Balanced but sometimes incoherent
```python
generate_text("Lily got a new puppy", temperature=0.6)

"""
Lily got a new puppy named Spot. He hit his ball out and fell on the floor. 
He cried and benly, but Spot was too fit fast. 
His friend cheered and then flew to the ball. 
They pulled and tugged and tugged. They ran away.
Spot gave the ball a kiss and the ball back to their mommy.
...
"""
```

#### 🟥 Temperature 0.8 — Fluent but nonsense
```python
generate_text("Lily got a new puppy", temperature=0.8)

"""
Lily got a new puppy and stopped pretending,
because other automobiles walked through the fall.
Aftenma's mushy man purred a sleepy scene for a while, although Weggin circle. 
They barked and snuggled until Fridge was. 
Fred finally paddle of his bedroom were even blacker! 
Jeddy was so excited that he didn't want to give.
...
"""
```

---

## 🗃️ Data Pipeline

Fable includes a small data-preparation utility that downloads and processes the TinyStories dataset for training.

```bash
# Download, clean, and tokenise text data
fable-prepare-data --stage all

# Or run an individual stage (download, clean, tokenise)
fable-prepare-data --stage clean
```

This creates:
- `data/raw/` — raw TinyStories dataset `.txt` files.  
- `data/clean/` — cleaned text filtered to supported characters.  
- `data/tokenized/` — `int8` binary token buffers used for model training.  

While TinyStories is used by default, the same pipeline can be adapted for other small-scale narrative datasets with minimal modification.

---

## 🧠 Training

Once data is prepared, train a model from scratch using either Python or the CLI.

### Python API

```python
from fable import save, train

model = train()
save(model)
```

### Command-Line Interface

```bash
fable-train --num-epochs 5 --batch-size 128 --learning-rate 3e-4
```

Training progress and checkpoints are saved automatically in `checkpoints/`.

---

## 🏗️ Project Structure

```
fable/
├── checkpoint.py              # Orbax save/load wrappers for NNX state trees
├── config.py                  # GPTConfig dataclass and defaults
├── data/
│   ├── pipeline.py            # Text data download/clean/tokenise commands
│   ├── tokenize.py            # Character-level tokenizer and vocabulary tools
│   └── tokenizer-config.json  # Vocabulary and end-of-text token definitions
├── evaluate.py                # Validation step shared across training and notebooks
├── generate.py                # Text generation helpers and CLI entry point
├── model/
│   ├── attention.py           # Multi-head self-attention
│   ├── dropout.py             # Lightweight stochastic dropout layer
│   ├── feed_forward.py        # GELU MLP block
│   ├── gpt.py                 # GPT model assembly and forward pass
│   ├── normalize.py           # Layer normalisation layer
│   ├── position.py            # Sinusoidal positional embeddings
│   └── transformer.py         # Pre-norm decoder block with dropout
├── train.py                   # JIT-compiled training loop with Optax optimizers
└── utils.py                   # TQDM helper wrappers

checkpoints/
└── demo.ckpt/                 # Example checkpoint trained for ~2 hours on RTX 4090
```

---

## 📚 Citation

If Fable supports your research or teaching, please cite:

```bibtex
@software{fable2025,
  title = {Fable: A Compact Storytelling Language Model in JAX},
  author = {Alex Goddard},
  year = {2025},
  url = {https://github.com/auxeno/fable}
}
```

---

## 📜 License

Released under the MIT License.  
See [`licence`](licence) for the full text.

---

## 🌟 Acknowledgments

Thanks to the creators of the TinyStories dataset (Eldan & Li),  
and to the JAX and Flax contributors whose work made Fable possible.
