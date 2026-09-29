import csv
import os
import subprocess
import time
from datetime import datetime

import psutil

LOG_FILE = "pi_health_log.csv"
INTERVAL_SECONDS = 2


def run_command(command):
    try:
        return subprocess.check_output(
            command,
            shell=True,
            text=True
        ).strip()
    except Exception:
        return "ERROR"


def get_temperature():
    output = run_command("vcgencmd measure_temp")

    try:
        return float(output.replace("temp=", "").replace("'C", ""))
    except ValueError:
        return None


def get_throttled_raw():
    return run_command("vcgencmd get_throttled")


def decode_throttled(raw):
    try:
        value = int(raw.split("=")[1], 16)
    except Exception:
        return {
            "undervoltage_now": None,
            "frequency_capped_now": None,
            "throttled_now": None,
            "undervoltage_occurred": None,
            "frequency_capped_occurred": None,
            "throttling_occurred": None,
        }

    return {
        "undervoltage_now": bool(value & (1 << 0)),
        "frequency_capped_now": bool(value & (1 << 1)),
        "throttled_now": bool(value & (1 << 2)),
        "undervoltage_occurred": bool(value & (1 << 16)),
        "frequency_capped_occurred": bool(value & (1 << 17)),
        "throttling_occurred": bool(value & (1 << 18)),
    }


def camera_connected():
    return os.path.exists("/dev/video0")


file_exists = os.path.exists(LOG_FILE)

with open(LOG_FILE, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "temperature_c",
            "throttled_raw",
            "undervoltage_now",
            "frequency_capped_now",
            "throttled_now",
            "undervoltage_occurred",
            "frequency_capped_occurred",
            "throttling_occurred",
            "cpu_percent",
            "cpu_frequency_mhz",
            "memory_percent",
            "memory_used_mb",
            "camera_connected"
        ])

    print("Pi health logger running.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            timestamp = datetime.now().isoformat(timespec="seconds")

            temperature = get_temperature()

            throttled_raw = get_throttled_raw()
            throttle = decode_throttled(throttled_raw)

            cpu_percent = psutil.cpu_percent(interval=None)

            frequency = psutil.cpu_freq()
            cpu_frequency = frequency.current if frequency else None

            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)

            camera = camera_connected()

            writer.writerow([
                timestamp,
                temperature,
                throttled_raw,
                throttle["undervoltage_now"],
                throttle["frequency_capped_now"],
                throttle["throttled_now"],
                throttle["undervoltage_occurred"],
                throttle["frequency_capped_occurred"],
                throttle["throttling_occurred"],
                cpu_percent,
                cpu_frequency,
                memory.percent,
                round(memory_used_mb, 1),
                camera
            ])

            file.flush()

            print(
                f"{timestamp} | "
                f"{temperature} C | "
                f"CPU {cpu_percent}% | "
                f"RAM {memory.percent}% | "
                f"UV now: {throttle['undervoltage_now']} | "
                f"UV occurred: {throttle['undervoltage_occurred']} | "
                f"Camera: {camera}"
            )

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nLogger stopped.")
        print(f"Saved to {LOG_FILE}")
