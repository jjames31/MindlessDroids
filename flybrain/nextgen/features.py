"""Shared actor observation schema used by simulation and Gazebo deployment."""
import math
import numpy as np

from .config import OBSERVATION_SIZE

FEATURE_NAMES = (
    "left_motion", "right_motion", "wide_motion", "radial_looming",
    "vertical_visual_bias", "banc_left", "banc_right", "banc_power",
    "banc_looming", "banc_looming_left", "banc_looming_right",
    "banc_vertical_escape", "forward_velocity", "lateral_velocity",
    "vertical_velocity", "sin_goal_bearing", "cos_goal_bearing",
    "goal_distance", "altitude", "sin_yaw", "cos_yaw", "roll", "pitch",
    "mass", "footprint", "vertical_radius", "propulsion", "max_pitch",
    "camera_fov", "previous_speed", "previous_yaw", "previous_climb",
)
assert len(FEATURE_NAMES) == OBSERVATION_SIZE

SCALE = np.asarray([
    8, 8, 8, 2, 2, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 1, .08, .25, 1, 1, 1.5, 1.5,
    .25, 1.5, 3, 1, .04, .5, 2, 1, 1,
], dtype=np.float32)


def actor_observation(*, left=0., right=0., wide=0., radial=0., vertical_bias=0.,
                      banc=None, velocity=(0., 0., 0.), yaw=0., roll=0., pitch=0.,
                      position=(0., 0., 3.), goal=(8., 0.), profile=None,
                      previous_action=(0., 0., 0.)):
    banc = banc or {}
    profile = profile or {}
    north, east, altitude = map(float, position)
    desired = math.atan2(float(goal[1])-east, float(goal[0])-north)
    bearing = math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
    distance = math.hypot(float(goal[0])-north, float(goal[1])-east)
    values = np.asarray([
        left, right, wide, radial, vertical_bias,
        banc.get("steering_left", 0.), banc.get("steering_right", 0.),
        banc.get("power", 0.), banc.get("looming", 0.),
        banc.get("looming_left", 0.), banc.get("looming_right", 0.),
        banc.get("vertical_escape", 0.), velocity[0], velocity[1], velocity[2],
        math.sin(bearing), math.cos(bearing), distance, altitude,
        math.sin(yaw), math.cos(yaw), roll, pitch,
        profile.get("takeoff_mass_kg", profile.get("mass_kg", 2.5)),
        profile.get("rotor_footprint_radius_m", .48),
        profile.get("vertical_clearance_radius_m", .22),
        profile.get("propulsion_scale", 1.),
        profile.get("maximum_pitch_deg", 30.),
        profile.get("camera_horizontal_fov_rad", 1.8),
        previous_action[0], previous_action[1], previous_action[2],
    ], dtype=np.float32)
    return np.clip(values * SCALE, -5., 5.)



def ned_to_heading_velocity(vn, ve, vd, yaw):
    """NED ground velocity to heading-forward/right and vertical-up (m/s).

    This is a level heading frame, not the roll/pitch-tilted aircraft frame.
    LOCAL_POSITION_NED is earth-fixed; using vx/vy directly is wrong after yaw.
    """
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return (c*float(vn)+s*float(ve), -s*float(vn)+c*float(ve), -float(vd))
