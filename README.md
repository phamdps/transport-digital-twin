# 🚦 Multimodal Traffic Prediction via Shared Semantic Token Space (MLLM-Traffic)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *"Recently, Multimodal Large Language Models (MLLMs) have introduced a new paradigm by mapping multimodal inputs into a shared semantic token space, enabling knowledge injection and unified generative reasoning for cross-modal and cross-task modeling."*

**MLLM-Traffic** is an open-source framework for urban mobility prediction that eliminates handcrafted fusion schemes and task-specific modular architectures. By converting continuous sensor time-series, non-Euclidean road graphs, visual feeds, and natural language context into a unified high-dimensional token space ($d_{\text{llm}}$), it enables a single frozen LLM backbone to perform multi-task forecasting, missing data imputation, anomaly reasoning, and counterfactual scenario simulations.

---

## 🏗 System Architecture & Paradigm Mapping

The repository structure directly operationalizes the four core pillars of the MLLM paradigm:


```

+-----------------------------------------------------------------------------------+
| 1. MULTIMODAL INPUT MAPPING (`models/tokenizers/` & `data/loaders/`)              |
|  [Sensor Time-Series]   [Road Graph Topology]   [CCTV Visual Feeds]   [Text Logs] |
+-------------------+-------------------+-------------------+-------------------+
|                   |                   |
v                   v                   v
+-----------------------------------------------------------------------------------+
| 2. SHARED SEMANTIC TOKEN SPACE (`models/space/`)                                  |
|  [ Temporal Patches ] + [ Graph LapPE Encodings ] + [ Latent Alignment Projection ]|
+---------------------------------------+-------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
| 3. KNOWLEDGE INJECTION (`models/injection/`)                                      |
|  Interleaving text context (Weather, Accidents, Events) + LoRA Domain Adapters    |
+---------------------------------------+-------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
| 4. UNIFIED GENERATIVE REASONING (`models/backbone/` & `tasks/`)                   |
|  Cross-Modal & Cross-Task Generation: Forecasting, Imputation, Root-Cause Analysis|
+-----------------------------------------------------------------------------------+

```

---

## 🛠 Project Structure


```

multimodal-traffic-mllm/
├── config/                         # Configuration management
│   ├── model_config.yaml           # LLM backbone & token embedding dimensions (d_llm)
│   ├── tokenizers.yaml             # Patch size, LapPE dimensions, vision encoders
│   └── tasks/                      # Task-specific prompts & target parameters
│       ├── forecasting.yaml        # Generative spatio-temporal forecasting specs
│       ├── imputation.yaml         # Masked token reconstruction parameters
│       └── root_cause.yaml         # Anomaly explanation & incident analysis specs
│
├── data/                           # Data loading & preprocessing pipelines
│   ├── loaders/                    # Modality-specific data ingestion
│   │   ├── sensor_loader.py        # Loop detector, GPS, and floating-car time series
│   │   ├── graph_loader.py         # Road network adjacency & topology loading
│   │   ├── vision_loader.py        # CCTV and camera feed preprocessors
│   │   └── text_loader.py          # Weather, accident logs, and event text alerts
│   └── processors/                 # Spatiotemporal alignment & masking utilities
│       ├── spatial_aligner.py      # Map spatial coordinates to graph node IDs
│       └── masking_utils.py        # Dynamic mask generation for missing sensor data
│
├── models/                         # Model components & embedding architecture
│   ├── tokenizers/                 # 1. MULTIMODAL INPUT MAPPING TO TOKENS
│   │   ├── temporal_patch.py       # 1D-Conv patchification for continuous time-series
│   │   ├── graph_tokenizer.py      # Laplacian Positional Encodings (LapPE) + ST-GNN
│   │   └── vision_tokenizer.py     # ViT patch projection heads
│   │
│   ├── space/                      # 2. SHARED SEMANTIC TOKEN SPACE
│   │   ├── latent_aligner.py       # Linear/MLP projection heads to d_llm
│   │   ├── positional_embeds.py    # Time-of-day, day-of-week & spatial PE fusion
│   │   └── contrastive_loss.py     # InfoNCE & physics alignment loss modules
│   │
│   ├── injection/                  # 3. KNOWLEDGE INJECTION MODULES
│   │   ├── prompt_builder.py       # Interleaving context text + spatial-temporal tokens
│   │   └── lora_adapters.py        # Parameter-efficient adapters for domain injection
│   │
│   └── backbone/                   # 4. UNIFIED GENERATIVE REASONING ENGINE
│       ├── mllm_wrapper.py         # Frozen LLM wrapper (Llama-3 / Qwen-2 / Mistral)
│       └── unified_decoder.py      # Autoregressive generation for cross-modal output
│
├── tasks/                          # 5. CROSS-TASK & CROSS-MODAL EXECUTORS
│   ├── forecast_executor.py        # Multi-step speed/flow generative forecasting
│   ├── imputation_executor.py      # Reconstructing missing sensor values via LLM context
│   ├── incident_reasoner.py        # Root-cause analysis & natural language explanations
│   └── counterfactual_sim.py       # "What-if" scenario simulation runner
│
├── scripts/                        # Training & evaluation routines
│   ├── train_stage1_align.py       # Stage 1: Pre-training token projection alignment
│   ├── train_stage2_instruct.py    # Stage 2: Instruction fine-tuning (LoRA)
│   └── evaluate_cross_task.py      # Unified evaluation across forecasting & reasoning
│
├── notebooks/                      # Interactive tutorials & demos
│   ├── 01_token_space_viz.ipynb    # Visualizing cross-modal alignment in latent space
│   └── 02_cross_task_demo.ipynb    # End-to-end inference across multiple tasks
│
├── requirements.txt                # Dependencies (PyTorch, PyG, Transformers, PEFT)
├── setup.py                        # Package installation script
└── README.md                       # Main project documentation

```

---

## ⚡ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/phamdps/multimodal-traffic-mllm.git](https://github.com/phamdps/multimodal-traffic-mllm.git)
cd multimodal-traffic-mllm

# Create conda environment
conda create -n mllm_traffic python=3.10 -y
conda activate mllm_traffic

# Install dependencies
pip install -r requirements.txt
pip install -e .

```

### 2. Pre-stage Data Preparation

Prepare benchmark datasets (e.g., PeMS-BAY, METR-LA) and compute graph Laplacian embeddings:

```bash
python data/processors/spatial_aligner.py --dataset METR-LA --output_dir ./data/processed/

```

### 3. Usage Example

Execute unified cross-modal inference using the shared semantic space:

```python
from models.backbone.mllm_wrapper import MLLMBackbone
from models.injection.prompt_builder import MultimodalPromptBuilder
from data.loaders.sensor_loader import LoadSensorStream
from data.loaders.graph_loader import LoadRoadGraph

# 1. Initialize MLLM Backbone & Token Projection Space
model = MLLMBackbone.from_pretrained("config/model_config.yaml")

# 2. Ingest Multimodal Physical Inputs
sensor_tokens = LoadSensorStream("./data/processed/pems_speed.pt")
graph_tokens = LoadRoadGraph("./data/processed/metr_la_lappe.pkl")

# 3. Inject Context & Construct Multimodal Prompt Sequence
prompt_builder = MultimodalPromptBuilder()
inputs = prompt_builder.build(
    text_context="Weather: Heavy Rain. Incident: Vehicle breakdown at Junction 12.",
    sensor_data=sensor_tokens,
    graph_topology=graph_tokens,
    task="forecasting_and_explanation"
)

# 4. Perform Unified Generative Reasoning
output = model.generate(inputs, max_new_tokens=256)
print("Unified Cross-Modal Result:")
print(output)

```

---

## 🏋️ Training Pipeline

The framework uses a two-stage training paradigm:

1. **Stage 1: Modality Alignment (`scripts/train_stage1_align.py`)**
Trains the linear/MLP projection heads in `models/space/latent_aligner.py` using contrastive alignment (InfoNCE) to map sensor time-series patches and graph LapPE embeddings into the LLM embedding space ($d_{\text{llm}}$).
2. **Stage 2: Instruction Tuning (`scripts/train_stage2_instruct.py`)**
Fine-tunes Low-Rank Adaptation (LoRA) adapters (`models/injection/lora_adapters.py`) on instruction datasets to perform joint forecasting, imputation, and natural language reasoning.

---

## 📄 License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).
