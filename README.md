# Transportation Digital Twin


## 1. Introduction

Extreme weather events and natural disasters, such as floods and hurricanes, are increasingly disrupting critical infrastructure systems worldwide. Urban transportation networks, which are essential for evacuations, delivering emergency resources, and maintaining connectivity during crises, are particularly vulnerable. Recent hurricanes, which have caused widespread disruptions along the southern states and the U.S. East Coast, have underscored the fragility of these systems.

Effectively planning for resilience and managing disruptions to transportation systems requires the use of computationally intensive transportation network models, which are often too resource-heavy and not scalable for real-time multi-scenario decision-making during disasters or for planning across regional networks. AI-powered digital twins for transportation networks offer a viable and effective tool to model and predict dynamic changes in mobility conditions, and support emergency planning and effective response

## 2. Background & Challenges

During disaster events, transportation networks experience rapid shifts in conditions due to flooded roads, severe traffic congestion, or damaged infrastructure. Traditional approaches for network modeling tasks—such as shortest path estimation and traffic assignment—struggle to capture these dynamic changes across multiple scenarios. Because these legacy methods are computationally intensive and rely heavily on static data, their lack of real-time adaptability leads to delays in emergency responses, inefficient evacuation routes, and heightened risks to public safety.

Key challenges include:

* **⚡ Computational Inefficiency:** Traditional routing and traffic assignment algorithms struggle to scale efficiently for large urban networks and multi-disruption scenarios, requiring significant processing time and delaying critical decision-making.
* **🌊 Dynamic Disruptions:** Disasters introduce frequent, unpredictable changes—such as flooding, accidents, or debris—causing network topology to change rapidly and rendering static models ineffective.
* **📡 Limited Integration of Real-Time Data:** Legacy transportation models lack the architecture to ingest live telemetry from diverse sources (e.g., traffic sensors, drones, satellite feeds) to reflect current ground-truth conditions.

---

## 💡 Proposed Solution & Core Capabilities

To address these challenges, we developed an **AI-powered Digital Twin for Transportation Networks** using **Graph Neural Networks (GNNs)** to model and predict mobility conditions amidst dynamic network changes. 

As a virtual replica of the physical transportation system, the digital twin mirrors real-world behavior by continuously fusing static infrastructure graphs with real-time sensor streams. It equips asset managers and emergency planners with real-time impact assessments, interactive monitoring, and scenario optimization for rapid, data-driven disaster response.

### Core Capabilities

* **🛣️ Shortest Path Estimation:** Leverages GNNs to predict optimal routes across dynamically changing networks. By utilizing node and edge embeddings that incorporate features like road lengths and live flood levels, the system updates shortest paths in real time without heavy computational overhead.
* **🚦 Dynamic Traffic Assignment (DTA):** Dynamically assigns traffic loads based on shifting demand and real-time network conditions, offering actionable insights into emerging congestion patterns and optimal rerouting options.
* **📡 Real-Time Data Fusion:** Ingests remote sensing data (e.g., satellite flood maps, drone feeds) and IoT sensor streams to continuously update network topology and reflect current hazards accurately.
* **🔮 Scenario Simulation & Resilience Modeling:** Simulates severe hazard scenarios (e.g., hurricanes, localized flooding) to evaluate network resilience, stress-test infrastructure vulnerabilities, and optimize evacuation strategies before impact.

## 3. Directory Structure

Create these files and folders in your project:

```text
trans_digital_twin/
├── .github/
│   └── ISSUE_TEMPLATE/
├── docs/
│   └── architecture.md
├── src/
│   ├── ingestion/       # Real-time IoT / GTFS data stream pipelines
│   ├── simulation/      # SUMO / SUMO-GUI traffic models
│   ├── twin_engine/     # Graph-based network model & state updates
│   └── visualization/   # 3D spatial UI (Deck.gl / Three.js / Cesium)
├── tests/
├── .gitignore
├── LICENSE
└── README.md

```

---

## 4. Quick Start

### Prerequisites

* **Python:** `3.10` or higher
* **Node.js:** `v18+` (for the web visualizer)
* **Docker & Docker Compose** (for Kafka and spatial databases)

### 1. Clone the Repository

```bash
git clone [https://github.com/phamdps/trans_digital_twin.git](https://github.com/phamdps/trans_digital_twin.git)
cd trans_digital_twin

```

### 2. Start Infrastructure

Launch Kafka, PostgreSQL/PostGIS, and Redis services:

```bash
docker-compose up -d

```

### 3. Set Up Python Environment

```bash

python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt

```

### 4. Run the Digital Twin Engine

```bash
python src/twin_engine/main.py --config config/dev.yaml

```

---

## 📊 Configuration

Configuration settings are stored in `config/default.yaml`. Update your API keys (Mapbox tile tokens, GTFS feed endpoints) before launching:

```yaml
telemetry:
  gtfs_rt_url: "[https://api.yourcity.gov/gtfs-rt/positions](https://api.yourcity.gov/gtfs-rt/positions)"
  poll_interval_sec: 5

visualization:
  mapbox_token: "YOUR_MAPBOX_ACCESS_TOKEN"
  initial_viewport:
    latitude: 48.8566
    longitude: 2.3522
    zoom: 12

```

---

## 🤝 Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](https://www.google.com/search?q=docs/CONTRIBUTING.md) for guidelines on branch naming, pull requests, and coding standards.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.