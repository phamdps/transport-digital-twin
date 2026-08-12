import os
import json
import time
import random
import sqlite3
import datetime
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
REALTIME_DIR = BASE_DIR / "data" / "realtime"
REALTIME_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = REALTIME_DIR / "latest_vehicles.json"
DB_PATH = REALTIME_DIR / "telemetry_history.db"
CONTROL_PATH = REALTIME_DIR / "control_signals.json"

# Initialize SQLite Telemetry Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_telemetry (
            vehicle_id TEXT,
            timestamp REAL,
            latitude REAL,
            longitude REAL,
            speed_kmh REAL,
            zone TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Initialize simulated vehicle fleet around San Francisco coordinates
vehicles = []
zones_list = ["Market Street Corridor", "Embarcadero South", "Financial District", "Mission District"]

for i in range(1, 41):
    vehicles.append({
        "vehicle_id": f"SF_FLEET_{i:03d}",
        # Centered around San Francisco downtown/Financial District
        "latitude": 37.7749 + random.uniform(-0.015, 0.015),
        "longitude": -122.4194 + random.uniform(-0.015, 0.015),
        "speed_kmh": random.uniform(12.0, 45.0),
        "zone": random.choice(zones_list)
    })

print(f"🚀 Starting SF Transport Telemetry Simulation with {len(vehicles)} active units...")
print(f"📡 Streaming updates to: {JSON_PATH}")

while True:
    current_time = time.time()
    
    # Check if Level 5 autonomous control signals are active
    active_caps = {}
    if CONTROL_PATH.exists():
        try:
            with open(CONTROL_PATH, "r") as cf:
                cdata = json.load(cf)
                if cdata.get("status") == "ACTIVE":
                    for intervention in cdata.get("interventions", []):
                        active_caps[intervention["target_zone"]] = intervention.get("speed_cap_kmh", 20.0)
        except:
            pass

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    updated_vehicles = []
    for v in vehicles:
        # Simulate movement
        v["latitude"] += random.uniform(-0.0004, 0.0004)
        v["longitude"] += random.uniform(-0.0004, 0.0004)
        
        base_speed = random.uniform(10.0, 45.0)
        
        # Apply closed-loop control speed cap if vehicle is in a restricted zone
        if v["zone"] in active_caps:
            speed_cap = active_caps[v["zone"]]
            v["speed_kmh"] = min(base_speed, speed_cap)
            print(f"⚠️ [CLOSED-LOOP ENFORCED] {v['vehicle_id']} in {v['zone']} capped at {v['speed_kmh']:.1f} km/h")
        else:
            v["speed_kmh"] = base_speed

        v_record = {
            "vehicle_id": v["vehicle_id"],
            "timestamp": current_time,
            "latitude": v["latitude"],
            "longitude": v["longitude"],
            "speed_kmh": round(v["speed_kmh"], 2),
            "zone": v["zone"]
        }
        updated_vehicles.append(v_record)

        # Log into SQLite history database
        cursor.execute("""
            INSERT INTO vehicle_telemetry (vehicle_id, timestamp, latitude, longitude, speed_kmh, zone)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (v_record["vehicle_id"], v_record["timestamp"], v_record["latitude"], v_record["longitude"], v_record["speed_kmh"], v_record["zone"]))

    conn.commit()
    conn.close()

    # Write latest JSON snapshot for Streamlit live consumption
    with open(JSON_PATH, "w") as f:
        json.dump(updated_vehicles, f, indent=4)

    time.sleep(3)