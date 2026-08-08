# Multimodal Transportation Digital Twin

A progressive, 5-level digital twin framework for modeling, monitoring, simulating, and optimizing complex urban transportation networks.

---

## 🏗️ Project Maturity Roadmap

This project evolves through 5 distinct digital twin maturity levels:

1. **Level 1: Descriptive / Monitoring Digital Twin** (`/level-1-descriptive`)
   - Static spatial network, OpenStreetMap integration, and infrastructure mapping.
2. **Level 2: Informative / Real-Time Operational Twin** (`/level-2-realtime`)
   - Live data ingestion (GTFS-Realtime, traffic APIs) and real-time operational dashboards.
3. **Level 3: Predictive Digital Twin** (`/level-3-predictive`)
   - Machine learning models for ETA forecasting, passenger demand prediction, and congestion trends.
4. **Level 4: Comprehensive Digital Twin** (`/level-4-comprehensive`)
   - Micro/meso-scopic traffic simulation (SUMO/CityFlow), scenario analysis, and disruption modeling.
5. **Level 5: Autonomous / Adaptive Digital Twin** (`/level-5-autonomous`)
   - Closed-loop optimization, dynamic signal control, and automated traffic management strategies.

---

## 📁 Repository Structure

```text
transport-digital-twin/
├── data/                       # Shared datasets and GIS maps
│   ├── raw/                    # Raw unprocessed data (GTFS, OSM extracts)
│   └── processed/              # Cleaned and processed datasets
├── docs/                       # Architecture diagrams and specifications
├── level-1-descriptive/        # Level 1: Static spatial model & baseline visualization
├── level-2-realtime/           # Level 2: Real-time streams & dashboard
├── level-3-predictive/         # Level 3: ML forecasting models
├── level-4-comprehensive/      # Level 4: Comprehensive simulation & scenario twin
├── level-5-autonomous/         # Level 5: Automated optimization & feedback loops
├── scripts/                    # Reusable utility scripts
└── README.md                   # Project overview and instructions
