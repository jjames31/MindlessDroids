from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PROFILE_PATH = PROJECT / "drone_profiles.json"
OBSERVATION_SIZE = 32
PRIVILEGED_SIZE = 40
ACTION_SIZE = 3
PROFILE_IDS = ("agile_330", "utility_450", "x550", "heavy_lift_650")
ACTION_NAMES = ("forward_speed", "yaw_rate", "climb_rate")
MAX_SPEED_MPS = 1.0
MAX_YAW_RPS = 1.2
MAX_CLIMB_MPS = 1.0


# Stable identifiers remain available for historical and optional stress reports.
REQUIRED_PROFILE_IDS = ("agile_330", "utility_450", "x550")
STRESS_PROFILE_IDS = ("heavy_lift_650",)
DEFAULT_TRAINING_ENVS = 1023
DEFAULT_EVAL_ENVS = 126
DEFAULT_HOLDOUT_EPISODES = 3000
DEFAULT_DEVELOPMENT_EPISODES = 384


def normalize_profiles(profiles=None):
    if profiles is None:
        profiles = REQUIRED_PROFILE_IDS
    if isinstance(profiles, str):
        raise ValueError("profiles must be a sequence, not a string")
    selected = tuple(profiles)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select distinct, nonempty airframe profiles")
    if any(p not in PROFILE_IDS for p in selected):
        raise ValueError("unknown airframe profile")
    return selected


def balanced_size(requested, profiles=None):
    count = len(normalize_profiles(profiles))
    return max(count, int(requested)//count*count)
