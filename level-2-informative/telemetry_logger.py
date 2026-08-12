import os
import json
import time
import sqlite3
from pathlib import Path

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
REALTIME_DATA_DIR = BASE_DIR / "data" / "realtime"
DB_PATH = REALTIME_DATA_DIR / "telemetry_history.db"

def init_db():
    """Initializes the SQLite database and creates the telemetry table if it doesn't exist."""
    os.makedirs(REALTIME_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            vehicle_id TEXT,
            latitude REAL,
            longitude REAL,
            speed_kmh REAL
        )
    """)
    conn.commit()
    conn.close()

def run_logger():
    """
    Continuously reads the latest vehicle JSON stream and logs it into SQLite for historical analysis.
    """
    init_db()
    json_path = REALTIME_DATA_DIR / "latest_vehicles.json"
    
    print(f"🗄️ Initialized telemetry storage at {DB_PATH}")
    print("Listening for incoming vehicle snapshots. Press Ctrl+C to exit.\n")

    last_processed_timestamp = 0

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        while True:
            if json_path.exists():
                try:
                    with open(json_path, "r") as f:
                        telemetry_batch = json.load(f)
                    
                    if telemetry_batch:
                        current_ts = telemetry_batch[0]["timestamp"]
                        
                        # Only insert if it's a new batch
                        if current_ts > last_processed_timestamp:
                            records = [
                                (v["timestamp"], v["vehicle_id"], v["latitude"], v["longitude"], v["speed_kmh"])
                                for v in telemetry_batch
                            ]
                            cursor.executemany("""
                                INSERT INTO vehicle_telemetry (timestamp, vehicle_id, latitude, longitude, speed_kmh)
                                VALUES (?, ?, ?, ?, ?)
                            """, records)
                            conn.commit()
                            
                            last_processed_timestamp = current_ts
                            print(f"[{time.strftime('%H:%M:%S')}] Saved {len(records)} telemetry records to database.")
                except (json.JSONDecodeError, IOError):
                    # Handle minor file-read race conditions while the producer is writing
                    pass

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 Telemetry logger terminated safely.")
    finally:
        conn.close()

if __name__ == "__main__":
    run_logger()
