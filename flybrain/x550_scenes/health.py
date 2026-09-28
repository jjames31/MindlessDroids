"""Read-only resource checks for bounded simulation experiments."""
import shutil
import subprocess
from pathlib import Path


def resource_health(path, use_cuda):
    disk = shutil.disk_usage(path).free / 1024**3
    lines = Path('/proc/meminfo').read_text().splitlines()
    ram = next(float(x.split()[1])/1024**2 for x in lines if x.startswith('MemAvailable:'))
    result = {'free_disk_gib': disk, 'available_ram_gib': ram}
    if disk < 20 or ram < 1:
        raise RuntimeError('resource limit reached: ' + str(result))
    if use_cuda:
        query = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10, check=True)
        temperatures = [float(v) for v in query.stdout.splitlines() if v.strip()]
        if not temperatures or any(not (0 <= v < 80) for v in temperatures):
            raise RuntimeError('GPU telemetry missing or temperature limit reached')
        result['gpu_temperature_c'] = max(temperatures)
    return result
