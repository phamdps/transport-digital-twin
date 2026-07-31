# 🚗⚡ Transportation Digital Twin for Crisis Resilience (TDT-CR)

A **Multimodal Foundation Model (MFM)-powered Transportation Digital Twin** designed for real-time monitoring, out-of-distribution traffic forecasting, and emergency response optimization during severe network disruptions (e.g., severe weather, floodings, structural failures, cyber-attacks, and mass evacuations).

---

## 📌 Key Features

* **Multimodal Fusion:** Integrates high-frequency spatio-temporal sensor data (loop detectors, GPS probes) with unstructured text reports (911 dispatch logs, emergency alerts, social media) and geospatial imagery/radar.
* **Text-to-Graph Cross-Modal Attention (T2G-CMA):** Dynamically conditions physical road network topology and traffic graph representations on textual incident descriptions.
* **Geospatial Distance Biasing:** Grounds cross-attention mechanisms using Haversine distance priors, ensuring incident alerts selectively impact proximate network nodes while retaining global detour awareness.
* **Out-of-Distribution (OOD) & Zero-Shot Forecasting:** Predicts non-linear spillover dynamics and cascading congestion during unprecedented shock events without requiring retraining.
* **Gated Multi-Horizon Resilience Engine:** Computes critical crisis metrics including **Peak Delay Delay-Time (PDDT)**, **Recovery Time Accuracy (RTA)**, and **Cascading Horizon Error (CHHE)** for interactive "what-if" policy simulations.

---

## 🏗 System Architecture

```
                    ┌──────────────────────────────────────────┐
                    │          MULTIMODAL DATA INPUTS          │
                    │   • Spatio-Temporal Sensor Data          │
                    │   • Textual Incident & Disaster Alerts   │
                    │   • Geographic Graphs & Aerial Feeds     │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │    MULTIMODAL FOUNDATION MODEL (MFM)     │
                    │                                          │
                    │  ┌────────────────────────────────────┐  │
                    │  │ Spatial-Temporal Graph Encoder     │  │
                    │  └─────────────────┬──────────────────┘  │
                    │                    ▼                     │
                    │  ┌────────────────────────────────────┐  │
                    │  │ Text-to-Graph Cross Attention      │  │
                    │  │ (with Geo-Distance Bias)           │  │
                    │  └─────────────────┬──────────────────┘  │
                    │                    ▼                     │
                    │  ┌────────────────────────────────────┐  │
                    │  │ Gated Residual Fusion Layer        │  │
                    │  └────────────────────────────────────┘  │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │       DIGITAL TWIN SIMULATION HUB        │
                    │  • Multi-Horizon Forecasting Engine      │
                    │  • Counterfactual "What-If" Analysis     │
                    │  • Dynamic Evacuation & Route Guidance   │
                    └──────────────────────────────────────────┘

```

---

## 🧮 Mathematical Core

The core cross-modal attention module conditions spatio-temporal node features $\mathbf{H}_{ST} \in \mathbb{R}^{N \times d_m}$ on contextual textual embeddings $\mathbf{H}_{Text} \in \mathbb{R}^{L \times d_m}$ using a spatially-biased multi-head attention formulation:

$$\mathbf{S}_{\text{Biased}}^{(h)} = \frac{\mathbf{Q}_{ST}^{(h)} (\mathbf{K}_{Text}^{(h)})^T}{\sqrt{d_k}} + \mathbf{B}_{Geo}$$

$$\mathbf{A}^{(h)} = \text{Softmax}\left(\mathbf{S}_{\text{Biased}}^{(h)}\right)$$

$$\mathbf{H}_{ST}' = \text{LayerNorm}\left( \mathbf{H}_{ST} + \mathbf{\gamma} \odot \text{MHCA}(\mathbf{H}_{ST}, \mathbf{H}_{Text}) \right)$$

where $\mathbf{B}_{Geo}$ applies continuous spatial distance penalties to prevent ungrounded long-range attention allocations, and $\mathbf{\gamma}$ is an adaptive gating vector balancing historical baseline features with emergency incident context.

---

## 🚀 Getting Started

### Prerequisites

* **Python:** $\ge 3.10$
* **CUDA:** $\ge 11.8$ / PyTorch $\ge 2.0$
* **Core Libraries:** `torch`, `torch-geometric`, `transformers`, `geopandas`, `networkx`

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/tdt-crisis-resilience.git
cd tdt-crisis-resilience

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```

---

## 💻 Quick Start & Usage

### 1. Ingest Multimodal Incident Data

```python
from tdt.models import MultimodalTrafficFM
from tdt.data import GraphDataLoader, TextIncidentProcessor

# Load spatial-temporal graph data and incident text
st_data = GraphDataLoader.load_network("data/network_graph.gml", lookback_steps=12)
incidents = TextIncidentProcessor.parse("Main St Bridge closed due to severe flooding near Junction 4.")

# Initialize Foundation Model
model = MultimodalTrafficFM.from_pretrained("tdt-fm-base-v1")

```

### 2. Predict Cascading Congestion

```python
# Run multimodal forward pass
forecast = model.predict_crisis_horizon(
    st_graph=st_data,
    text_context=incidents,
    forecast_horizon_steps=24 # e.g., 2 hours ahead
)

print(f"Predicted Peak Congestion Delay (PDDT): {forecast.pddt} mins")

```

---

## 📂 Project Structure

```
tdt-crisis-resilience/
├── configs/                  # Model architectures and hyperparameter configs
├── data/                     # Sample graph topologies, incidents, and dataset scripts
├── models/
│   ├── encoders/             # Spatial-Temporal Graph Encoders (STGNN/ST-Transformer)
│   ├── attention/            # T2G Cross-Modal Attention & Geo-Bias modules
│   └── decoder/              # Multi-horizon forecasting decoders
├── digital_twin/             # Digital Twin simulator & counterfactual scenario modules
├── metrics/                  # Resilience error functions (PDDT, RTA, CHHE)
├── tests/                    # Unit and integration test suites
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

```

---

## 📚 References & Citation

If you use this model framework or dataset structure in your research, please cite our foundational paper:

```bibtex
@article{tdt_crisis_resilience_2026,
  title={Multimodal Foundation Models for Transportation Digital Twins in Crisis Resilience},
  author={Your Research Team},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026}
}

```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.