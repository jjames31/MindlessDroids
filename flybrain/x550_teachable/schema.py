"""Named, versioned observation/action contract shared by training and runtime."""
import numpy as np
from nextgen.features import FEATURE_NAMES as LEGACY_FEATURE_NAMES

SCHEMA = 'x550-recurrent-depth-v2'
STATE_NAMES = tuple(LEGACY_FEATURE_NAMES[12:32])
DEPTH_NAMES = tuple(f'depth_r{r}_c{c}_metres_over_8' for r in range(3) for c in range(9))
VALIDITY_NAMES = tuple(f'depth_r{r}_c{c}_valid_fraction' for r in range(3) for c in range(9))
OBSERVATION_NAMES = STATE_NAMES + DEPTH_NAMES + VALIDITY_NAMES + (
    'observation_age_seconds_over_0_5', 'estimator_quality')
OBSERVATION_SIZE = len(OBSERVATION_NAMES)
VELOCITY_SLICE = slice(0, 3)
DEPTH_SLICE = slice(20, 47)
VALIDITY_SLICE = slice(47, 74)
AGE_INDEX = 74
QUALITY_INDEX = 75
ACTION_NAMES = ('forward_speed_normalized', 'yaw_rate_normalized', 'climb_rate_normalized')
ACTION_STAGES = ('sampled_raw', 'proposed_tanh', 'sensor_shielded',
                 'altitude_adapted', 'delay_applied')
AUXILIARY_NAMES = ('next_forward_velocity_mps', 'next_lateral_velocity_mps',
                   'next_vertical_velocity_mps', 'nearest_depth_delta_metres_over_4',
                   'collision_next_step', 'near_obstacle_next_step')

assert OBSERVATION_SIZE == 76
assert STATE_NAMES[:3] == ('forward_velocity', 'lateral_velocity', 'vertical_velocity')


def validate_observation(value):
    x = np.asarray(value, np.float32)
    if x.shape[-1] != OBSERVATION_SIZE or not np.isfinite(x).all():
        raise ValueError('invalid x550 recurrent observation')
    if ((x[..., DEPTH_SLICE] < 0) | (x[..., DEPTH_SLICE] > 1)).any():
        raise ValueError('normalized depth outside [0,1]')
    if ((x[..., VALIDITY_SLICE] < 0) | (x[..., VALIDITY_SLICE] > 1)).any():
        raise ValueError('depth validity outside [0,1]')
    if ((x[...,AGE_INDEX]<0)|(x[...,AGE_INDEX]>1)).any() \
            or ((x[...,QUALITY_INDEX]<0)|(x[...,QUALITY_INDEX]>1)).any():
        raise ValueError('observation age/quality outside [0,1]')
    if ((x[...,DEPTH_SLICE]==0)&(x[...,VALIDITY_SLICE]>0)).any():
        raise ValueError('zero/unknown depth cannot be marked valid')
    return x

