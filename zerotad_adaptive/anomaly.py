from __future__ import annotations


import math
from statistics import NormalDist
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .common import (
    EPS,
    AnomalyResult,
    Baseline,
    MotionSnapshot,
    RoadAxis,
    TrackState,
    VEHICLE_CLASS_IDS,
    adaptive_recent_count,
    equivalent_temporal_sample_count,
    clip01,
    combine_confidences,
    sample_reliability,
    weighted_mad,
    weighted_median,
    velocity_innovation,
)
from .road import RoadDirectionField, estimate_road_axis


REFERENCE_TAIL_PROBABILITY = 0.025
REFERENCE_Z = NormalDist().inv_cdf(1.0 - 0.5 * REFERENCE_TAIL_PROBABILITY)


ONE_SIDED_REFERENCE_Z = NormalDist().inv_cdf(1.0 - REFERENCE_TAIL_PROBABILITY)


PERSISTENT_FLOW_STOP_SCORE_CAP = 1.5


COLD_START_STOP_SCORE_CAP = 1.5


def project(vx: float, vy: float, axis: RoadAxis) -> Tuple[float, float]:
    perpendicular_x, perpendicular_y = -axis.y, axis.x
    return (
        float(vx * axis.x + vy * axis.y),
        float(vx * perpendicular_x + vy * perpendicular_y),
    )


def _weighted_source_statistics(
    values: Sequence[float],
    weights: Sequence[float],
) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    location = weighted_median(values, weights)
    scale = weighted_mad(values, weights, center=location)
    effective_weight = float(sum(max(0.0, float(weight)) for weight in weights))
    return float(location), float(scale), effective_weight


def empirical_bayes_baseline(
    prior_values: Sequence[float],
    prior_weights: Sequence[float],
    context_values: Sequence[float],
    context_weights: Sequence[float],
    default_mean: float,
    measurement_resolution: float,
    innovation_scale: float,
    anchor_mean: float | None = None,
    anchor_confidence: float = 0.0,
) -> Baseline:


    prior_location, prior_scale, prior_weight = _weighted_source_statistics(
        prior_values, prior_weights
    )
    context_location, context_scale, context_weight = _weighted_source_statistics(
        context_values, context_weights
    )
    anchor_weight = clip01(anchor_confidence) if anchor_mean is not None else 0.0

    weighted_locations: List[float] = []
    source_weights: List[float] = []
    if prior_weight > 0.0:
        weighted_locations.append(prior_location)
        source_weights.append(prior_weight)
    if context_weight > 0.0:
        weighted_locations.append(context_location)
        source_weights.append(context_weight)
    if anchor_weight > 0.0 and anchor_mean is not None:
        weighted_locations.append(float(anchor_mean))
        source_weights.append(anchor_weight)

    if source_weights:
        mean = float(np.average(weighted_locations, weights=source_weights))
    else:
        mean = float(default_mean)

    source_gap = 0.0
    if prior_weight > 0.0 and context_weight > 0.0:
        balance = math.sqrt(prior_weight * context_weight) / max(prior_weight + context_weight, EPS)
        source_gap = abs(prior_location - context_location) * balance

    sigma = max(
        float(measurement_resolution),
        float(innovation_scale),
        float(prior_scale),
        float(context_scale),
        float(source_gap),
        EPS,
    )

    prior_confidence = prior_weight / (1.0 + prior_weight)
    context_confidence = context_weight / (1.0 + context_weight)

    confidence = 1.0 - (
        (1.0 - prior_confidence)
        * (1.0 - context_confidence)
        * (1.0 - anchor_weight)
    )
    return Baseline(float(mean), float(sigma), clip01(confidence))


def score_from_z(z_value: float, confidence: float) -> float:


    effective_reference = REFERENCE_Z * math.sqrt(2.0 - clip01(confidence))
    return float(max(0.0, float(z_value)) / max(effective_reference, EPS))


def score_from_one_sided_z(z_value: float, confidence: float) -> float:


    effective_reference = ONE_SIDED_REFERENCE_Z * math.sqrt(
        2.0 - clip01(confidence)
    )
    return float(max(0.0, float(z_value)) / max(effective_reference, EPS))


def _effective_sample_size(weights: Sequence[float]) -> float:


    array = np.asarray(weights, dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    if array.size == 0:
        return 0.0
    return float(np.sum(array) ** 2 / max(np.sum(array * array), EPS))


def _weighted_standardized_sum(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:


    if not values:
        return 0.0
    array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(array)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
    )
    if not np.any(valid):
        return 0.0
    array = array[valid]
    weight_array = weight_array[valid]
    return float(
        np.sum(weight_array * array)
        / max(math.sqrt(float(np.sum(weight_array * weight_array))), EPS)
    )


def _update_stop_evidence_memory(
    state: TrackState,
    instantaneous_effective_z: float,
    observation_confidence: float,
    memory_window: int,
) -> Tuple[float, float, float, int]:


    window = max(2, int(memory_window))
    state.stop_z_history.append(max(0.0, float(instantaneous_effective_z)))
    state.stop_weight_history.append(clip01(observation_confidence))

    if len(state.stop_z_history) > window:
        state.stop_z_history = state.stop_z_history[-window:]
        state.stop_weight_history = state.stop_weight_history[-window:]

    temporal_z = _weighted_standardized_sum(
        state.stop_z_history,
        state.stop_weight_history,
    )
    effective_count = _effective_sample_size(state.stop_weight_history)
    total_weight = float(sum(max(0.0, value) for value in state.stop_weight_history))
    temporal_confidence = combine_confidences(
        [
            sample_reliability(int(round(effective_count))),
            total_weight / (1.0 + total_weight),
        ]
    )
    return (
        float(temporal_z),
        float(temporal_confidence),
        float(effective_count),
        int(len(state.stop_z_history)),
    )


def _update_persistent_flow_stop_memory(
    state: TrackState,
    instantaneous_flow_z: float,
    observation_confidence: float,
    memory_window: int,
    stable_near_stop: bool,
) -> Tuple[float, float, float, int]:


    if not stable_near_stop:
        state.persistent_flow_stop_z_history.clear()
        state.persistent_flow_stop_weight_history.clear()

    window = max(2, int(memory_window))
    state.persistent_flow_stop_z_history.append(
        max(0.0, float(instantaneous_flow_z))
    )
    state.persistent_flow_stop_weight_history.append(
        clip01(observation_confidence)
    )

    if len(state.persistent_flow_stop_z_history) > window:
        state.persistent_flow_stop_z_history = (
            state.persistent_flow_stop_z_history[-window:]
        )
        state.persistent_flow_stop_weight_history = (
            state.persistent_flow_stop_weight_history[-window:]
        )

    temporal_z = _weighted_standardized_sum(
        state.persistent_flow_stop_z_history,
        state.persistent_flow_stop_weight_history,
    )
    effective_count = _effective_sample_size(
        state.persistent_flow_stop_weight_history
    )
    total_weight = float(
        sum(
            max(0.0, value)
            for value in state.persistent_flow_stop_weight_history
        )
    )
    temporal_confidence = combine_confidences(
        [
            sample_reliability(int(round(effective_count))),
            total_weight / (1.0 + total_weight),
            observation_confidence,
        ]
    )
    return (
        float(temporal_z),
        float(temporal_confidence),
        float(effective_count),
        int(len(state.persistent_flow_stop_z_history)),
    )


def _update_lateral_evidence_memory(
    state: TrackState,
    signed_standardized_evidence: float,
    observation_confidence: float,
    memory_window: int,
    history_retention: float = 1.0,
) -> Tuple[float, float, float, float, int]:


    window = max(3, int(memory_window))


    retention = clip01(history_retention)
    if retention < 1.0 and state.lateral_signed_z_history:
        state.lateral_signed_z_history = [
            float(value) * retention
            for value in state.lateral_signed_z_history
        ]

    state.lateral_signed_z_history.append(float(signed_standardized_evidence))
    state.lateral_weight_history.append(clip01(observation_confidence))

    if len(state.lateral_signed_z_history) > window:
        state.lateral_signed_z_history = state.lateral_signed_z_history[-window:]
        state.lateral_weight_history = state.lateral_weight_history[-window:]

    z_values = np.asarray(state.lateral_signed_z_history, dtype=float)
    weights = np.asarray(state.lateral_weight_history, dtype=float)
    valid = (
        np.isfinite(z_values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        return 0.0, 0.0, 0.0, 0.0, len(state.lateral_signed_z_history)

    z_values = z_values[valid]
    weights = weights[valid]
    signed_sum = float(np.sum(weights * z_values))
    absolute_sum = float(np.sum(weights * np.abs(z_values)))
    directional_coherence = abs(signed_sum) / max(absolute_sum, EPS)


    effective_count = _effective_sample_size(weights.tolist())
    temporal_z = (
        abs(signed_sum)
        / max(
            math.sqrt(float(np.sum(weights * weights))),
            EPS,
        )
        * math.sqrt(
            max(
                0.0,
                sample_reliability(int(round(effective_count))),
            )
        )
    )

    total_weight = float(np.sum(weights))
    temporal_confidence = combine_confidences(
        [
            sample_reliability(int(round(effective_count))),
            total_weight / (1.0 + total_weight),
            directional_coherence,
        ]
    )
    return (
        float(temporal_z),
        float(temporal_confidence),
        float(directional_coherence),
        float(effective_count),
        int(len(state.lateral_signed_z_history)),
    )


def _motion_confidence(
    state: TrackState,
    motion: MotionSnapshot,
    historical_count: int,
    sampling_interval_scale: float = 1.0,
) -> Tuple[float, float, float, float]:
    recent_observations = state.observations[-min(4, len(state.observations)) :]
    detection_confidence = (
        float(np.median([observation.det_conf for observation in recent_observations]))
        if recent_observations
        else 0.0
    )
    track_confidence = sample_reliability(
        equivalent_temporal_sample_count(len(motion.vxs), sampling_interval_scale)
    )


    history_x = motion.vxs[:historical_count]
    history_y = motion.vys[:historical_count]
    historical_innovation = velocity_innovation(history_x, history_y)
    historical_speeds = [math.hypot(vx, vy) for vx, vy in zip(history_x, history_y)]
    historical_speed = float(np.median(historical_speeds)) if historical_speeds else motion.speed
    consistency = (
        historical_speed + motion.resolution
    ) / max(
        historical_speed + motion.resolution + historical_innovation,
        EPS,
    )
    return (
        detection_confidence,
        track_confidence,
        clip01(consistency),
        max(float(historical_innovation), float(motion.resolution)),
    )


def _current_projection(
    parallel: Sequence[float],
    perpendicular: Sequence[float],
    weights: Sequence[float],
) -> Tuple[float, float]:
    take = min(2, len(parallel))
    if take <= 0:
        return 0.0, 0.0
    current_parallel = weighted_median(parallel[-take:], weights[-take:])
    current_perpendicular = weighted_median(perpendicular[-take:], weights[-take:])
    return float(current_parallel), float(current_perpendicular)


def _window_weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not np.any(valid):
        return float(np.mean(array))
    return float(np.average(array[valid], weights=weight_array[valid]))


def _robust_track_stationarity(
    state: TrackState,
) -> Dict[str, object]:


    observations = state.observations[-min(12, len(state.observations)) :]
    if len(observations) < 2:
        return {
            "speed": 0.0,
            "noise": 1.0,
            "cluster_stability": 0.0,
            "duration_reliability": 0.0,
            "effective_count": 0.0,
            "elapsed_frames": 0.0,
            "latest_speeds": [],
            "latest_resolutions": [],
            "latest_weights": [],
        }

    pair_speeds: List[float] = []
    pair_resolutions: List[float] = []
    pair_weights: List[float] = []
    latest_speeds: List[float] = []
    latest_resolutions: List[float] = []
    latest_weights: List[float] = []
    latest = observations[-1]

    for left_index, previous in enumerate(observations[:-1]):
        for current in observations[left_index + 1 :]:
            dt = max(1, int(current.frame_id - previous.frame_id))
            scale = max(20.0, 0.5 * (previous.scale + current.scale))
            speed = math.hypot(
                current.ground_x - previous.ground_x,
                current.ground_y - previous.ground_y,
            ) / dt / scale
            resolution = 1.0 / (dt * scale)
            confidence = math.sqrt(
                max(0.0, previous.det_conf * current.det_conf)
            )


            baseline_reliability = math.sqrt(dt / (dt + 1.0))
            weight = confidence * baseline_reliability
            pair_speeds.append(float(speed))
            pair_resolutions.append(float(resolution))
            pair_weights.append(float(weight))

        dt_latest = max(1, int(latest.frame_id - previous.frame_id))
        scale_latest = max(20.0, 0.5 * (previous.scale + latest.scale))
        latest_speed = math.hypot(
            latest.ground_x - previous.ground_x,
            latest.ground_y - previous.ground_y,
        ) / dt_latest / scale_latest
        latest_resolution = 1.0 / (dt_latest * scale_latest)
        latest_weight = math.sqrt(max(0.0, previous.det_conf * latest.det_conf))
        latest_weight *= math.sqrt(dt_latest / (dt_latest + 1.0))
        latest_speeds.append(float(latest_speed))
        latest_resolutions.append(float(latest_resolution))
        latest_weights.append(float(latest_weight))

    robust_speed = weighted_median(pair_speeds, pair_weights)
    robust_noise = max(
        weighted_mad(pair_speeds, pair_weights, center=robust_speed),
        weighted_median(pair_resolutions, pair_weights),
        EPS,
    )

    observation_weights = [max(EPS, observation.det_conf) for observation in observations]
    anchor_x = weighted_median(
        [observation.ground_x for observation in observations],
        observation_weights,
    )
    anchor_y = weighted_median(
        [observation.ground_y for observation in observations],
        observation_weights,
    )
    normalized_radii = [
        math.hypot(
            observation.ground_x - anchor_x,
            observation.ground_y - anchor_y,
        )
        / max(20.0, observation.scale)
        for observation in observations
    ]
    cluster_radius = weighted_median(normalized_radii, observation_weights)
    localization_radius = weighted_median(
        [1.0 / max(20.0, observation.scale) for observation in observations],
        observation_weights,
    )
    cluster_stability = localization_radius / max(
        localization_radius + cluster_radius,
        EPS,
    )

    elapsed_frames = max(
        1.0,
        float(observations[-1].frame_id - observations[0].frame_id),
    )
    positive_dts = [
        max(1.0, float(current.frame_id - previous.frame_id))
        for previous, current in zip(observations[:-1], observations[1:])
    ]
    typical_dt = float(np.median(positive_dts)) if positive_dts else 1.0
    duration_reliability = elapsed_frames / max(
        elapsed_frames + typical_dt,
        EPS,
    )

    return {
        "speed": float(robust_speed),
        "noise": float(robust_noise),
        "cluster_stability": float(clip01(cluster_stability)),
        "duration_reliability": float(clip01(duration_reliability)),
        "effective_count": float(_effective_sample_size(pair_weights)),
        "elapsed_frames": float(elapsed_frames),
        "latest_speeds": latest_speeds,
        "latest_resolutions": latest_resolutions,
        "latest_weights": latest_weights,
    }


def _joint_motion_surprise(
    lateral_score: float,
    longitudinal_score: float,
    lateral_support: float,
) -> Tuple[float, float, float, float]:


    lateral = max(0.0, float(lateral_score))
    longitudinal = max(0.0, float(longitudinal_score))
    if lateral <= 0.0 or longitudinal <= 0.0:
        return (
            float(max(lateral, longitudinal)),
            0.0,
            0.0,
            float(max(lateral, longitudinal)),
        )

    joint_energy = math.hypot(lateral, longitudinal)
    balance = (
        2.0 * math.sqrt(lateral * longitudinal)
        / max(lateral + longitudinal, EPS)
    )


    gate = math.sqrt(clip01(lateral_support))
    strongest = max(lateral, longitudinal)
    joint_score = strongest + (joint_energy - strongest) * balance * gate
    return (
        float(joint_energy),
        float(balance),
        float(gate),
        float(joint_score),
    )


def _update_impact_episode_posterior(
    state: TrackState,
    instantaneous_score: float,
    continuation_support: float,
) -> Tuple[float, float]:


    current = max(0.0, float(instantaneous_score))
    support = clip01(continuation_support)


    if support <= EPS:
        state.impact_posterior_score = 0.0
        return 0.0, 0.0

    previous = max(0.0, float(state.impact_posterior_score))
    retention = math.sqrt(support)

    if previous <= current or previous <= EPS:
        posterior = current
    else:
        posterior = math.sqrt(
            max(
                0.0,
                (1.0 - retention) * current * current
                + retention * previous * previous,
            )
        )

    state.impact_posterior_score = float(posterior)
    return float(posterior), float(retention)


def compute_motion_anomaly(
    track_id: int,
    states: Dict[int, TrackState],
    motions: Dict[int, MotionSnapshot],
    neighbor_ids: Sequence[int],
    neighbor_distances: Dict[Tuple[int, int], float],
    road_field: RoadDirectionField,
    sampling_interval_scale: float = 1.0,
) -> AnomalyResult:
    state = states[track_id]
    motion = motions[track_id]
    sampling_interval_scale = max(1.0, float(sampling_interval_scale))


    temporal_evidence_scale = math.sqrt(sampling_interval_scale)
    empty = AnomalyResult(
        0.0,
        "NORMAL",
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        RoadAxis(),
        0,
        0.0,
        {


            "display_eligible": 0,
            "static_nonvehicle_suspect": 0,
        },
    )
    if state.class_id not in VEHICLE_CLASS_IDS or len(motion.vxs) < 2:
        return empty

    road_estimate = estimate_road_axis(
        track_id,
        states,
        motions,
        neighbor_ids,
        neighbor_distances,
        road_field,
        sampling_interval_scale=sampling_interval_scale,
    )
    axis = road_estimate.axis
    if not axis.ready:
        return empty

    parallel_sequence: List[float] = []
    perpendicular_sequence: List[float] = []
    for velocity_x, velocity_y in zip(motion.vxs, motion.vys):
        parallel, perpendicular = project(velocity_x, velocity_y, axis)
        parallel_sequence.append(parallel)
        perpendicular_sequence.append(perpendicular)

    current_parallel, current_perpendicular = _current_projection(
        parallel_sequence,
        perpendicular_sequence,
        motion.confidences,
    )

    recent_count = adaptive_recent_count(len(parallel_sequence), sampling_interval_scale)
    prior_end = max(0, len(parallel_sequence) - recent_count)
    prior_parallel = parallel_sequence[:prior_end]
    prior_perpendicular = perpendicular_sequence[:prior_end]
    prior_weights = motion.confidences[:prior_end]


    context_parallel: List[float] = []
    context_perpendicular: List[float] = []
    context_weights: List[float] = []


    stop_context_speeds: List[float] = []
    stop_context_weights: List[float] = []
    stop_context_resolutions: List[float] = []
    stop_context_corridor_memberships: List[float] = []
    stop_context_corridor_weights: List[float] = []


    used_context_track_weights: Dict[int, float] = {}
    used_stop_context_track_weights: Dict[int, float] = {}
    target_latest = state.observations[-1]
    perpendicular_axis = (-axis.y, axis.x)

    for neighbor_id, context_weight in road_estimate.context_weights.items():
        neighbor_motion = motions.get(neighbor_id)
        if (
            neighbor_motion is None
            or not neighbor_motion.ready
            or context_weight <= 0.0
        ):
            continue
        neighbor_parallel, neighbor_perpendicular = project(
            neighbor_motion.vx,
            neighbor_motion.vy,
            axis,
        )
        if neighbor_parallel <= 0.0:
            continue
        context_parallel.append(float(neighbor_parallel))
        context_perpendicular.append(float(neighbor_perpendicular))
        context_weights.append(float(context_weight))
        used_context_track_weights[int(neighbor_id)] = float(context_weight)

    for neighbor_id, context_weight in road_estimate.stop_context_weights.items():
        neighbor_motion = motions.get(neighbor_id)
        if (
            neighbor_motion is None
            or not neighbor_motion.ready
            or context_weight <= 0.0
        ):
            continue
        stop_context_speeds.append(float(neighbor_motion.speed))
        stop_context_weights.append(float(context_weight))
        stop_context_resolutions.append(float(neighbor_motion.resolution))
        used_stop_context_track_weights[int(neighbor_id)] = float(context_weight)


        neighbor_state = states.get(neighbor_id)
        if neighbor_state is not None and neighbor_state.observations:
            neighbor_latest = neighbor_state.observations[-1]
            relative_x = neighbor_latest.ground_x - target_latest.ground_x
            relative_y = neighbor_latest.ground_y - target_latest.ground_y
            lateral_gap_pixels = abs(
                relative_x * perpendicular_axis[0]
                + relative_y * perpendicular_axis[1]
            )
            pair_scale = max(
                20.0,
                0.5 * (target_latest.scale + neighbor_latest.scale),
            )
            normalized_lateral_gap = lateral_gap_pixels / pair_scale
            corridor_membership = 1.0 / (
                1.0 + normalized_lateral_gap * normalized_lateral_gap
            )
            stop_context_corridor_memberships.append(
                float(corridor_membership)
            )
            stop_context_corridor_weights.append(float(context_weight))

    default_parallel = (
        weighted_median(prior_parallel, prior_weights)
        if prior_parallel
        else current_parallel
    )
    (
        detection_confidence,
        track_confidence,
        motion_consistency,
        historical_innovation,
    ) = _motion_confidence(
        state, motion, prior_end, sampling_interval_scale=sampling_interval_scale
    )

    parallel_baseline = empirical_bayes_baseline(
        prior_parallel,
        prior_weights,
        context_parallel,
        context_weights,
        default_mean=default_parallel,
        measurement_resolution=motion.resolution,
        innovation_scale=historical_innovation,
    )

    perpendicular_baseline = empirical_bayes_baseline(
        prior_perpendicular,
        prior_weights,
        context_perpendicular,
        context_weights,
        default_mean=0.0,
        measurement_resolution=motion.resolution,
        innovation_scale=historical_innovation,
        anchor_mean=0.0,
        anchor_confidence=axis.confidence,
    )

    context_confidence = road_estimate.context_confidence
    stop_context_confidence = road_estimate.stop_context_confidence


    traffic_profile_query = getattr(road_field, "query_traffic_profile", None)
    if callable(traffic_profile_query):
        try:
            traffic_profile = traffic_profile_query(
                target_latest.ground_x,
                target_latest.ground_y,
                target_latest.bbox_xyxy,
            )
        except TypeError:
            traffic_profile = traffic_profile_query(
                target_latest.ground_x,
                target_latest.ground_y,
            )
    else:
        traffic_profile = {}
    historical_flow_speed = max(
        0.0,
        float(traffic_profile.get("speed", 0.0)),
    )
    historical_flow_noise = max(
        0.0,
        float(traffic_profile.get("speed_sigma", 0.0)),
    )
    historical_flow_confidence = clip01(
        float(traffic_profile.get("confidence", 0.0))
    )
    historical_geometry_confidence = clip01(
        float(traffic_profile.get("geometry_confidence", 0.0))
    )

    flow_corridor_membership = (
        _window_weighted_mean(
            stop_context_corridor_memberships,
            stop_context_corridor_weights,
        )
        if stop_context_corridor_memberships
        else 0.0
    )


    occupancy_query = getattr(road_field, "query_traffic_occupancy", None)
    traffic_occupancy_confidence = (
        clip01(occupancy_query(target_latest.ground_x, target_latest.ground_y))
        if callable(occupancy_query)
        else 1.0
    )


    historical_corridor_support = (
        historical_flow_confidence * historical_geometry_confidence
    )
    effective_flow_corridor_membership = max(
        clip01(flow_corridor_membership),
        clip01(historical_corridor_support),
    )
    current_flow_road_membership = math.sqrt(
        max(
            0.0,
            traffic_occupancy_confidence
            * clip01(flow_corridor_membership),
        )
    )
    cold_start_flow_road_membership = math.sqrt(
        max(
            0.0,
            traffic_occupancy_confidence
            * clip01(effective_flow_corridor_membership),
        )
    )


    flow_road_membership = current_flow_road_membership

    robust_stationarity = _robust_track_stationarity(state)
    robust_track_speed = float(robust_stationarity["speed"])
    robust_track_noise = float(robust_stationarity["noise"])
    robust_cluster_stability = float(
        robust_stationarity["cluster_stability"]
    )
    robust_duration_reliability = float(
        robust_stationarity["duration_reliability"]
    )
    robust_motion_identity = robust_track_speed / max(
        robust_track_speed + robust_track_noise,
        EPS,
    )


    stationary_peer_mass = 0.0
    stationary_peer_count = 0
    for neighbor_id in neighbor_ids:
        neighbor_motion = motions.get(int(neighbor_id))
        if neighbor_motion is None or not neighbor_motion.ready:
            continue
        normalized_distance = float(
            neighbor_distances.get((track_id, int(neighbor_id)), 1.0)
        )
        proximity = 1.0 / (1.0 + normalized_distance * normalized_distance)
        peer_staticness = neighbor_motion.resolution / max(
            neighbor_motion.resolution + neighbor_motion.speed,
            EPS,
        )
        contribution = proximity * clip01(peer_staticness)
        stationary_peer_mass += float(contribution)
        if contribution > 0.0:
            stationary_peer_count += 1
    cold_start_isolation_gate = 1.0 / (1.0 + stationary_peer_mass)


    current_vehicle_identity = current_flow_road_membership
    historical_vehicle_identity = clip01(
        historical_flow_confidence * historical_geometry_confidence
    )
    road_vehicle_identity = max(
        clip01(current_vehicle_identity),
        clip01(historical_vehicle_identity),
    )
    vehicle_candidate_confidence = max(
        clip01(robust_motion_identity),
        clip01(road_vehicle_identity),
    )
    static_nonvehicle_suspect = bool(
        robust_motion_identity < 0.5
        and road_vehicle_identity < 0.35
        and historical_geometry_confidence < 0.35
        and vehicle_candidate_confidence < 0.5 * max(detection_confidence, EPS)
    )


    recent_perpendicular = perpendicular_sequence[-recent_count:]
    recent_parallel = parallel_sequence[-recent_count:]
    recent_weights = motion.confidences[-recent_count:]
    recent_dts = motion.dts[-recent_count:]

    signed_lateral = _window_weighted_mean(recent_perpendicular, recent_weights)
    lateral_abs_mean = _window_weighted_mean(
        [abs(value) for value in recent_perpendicular], recent_weights
    )
    parallel_abs_mean = _window_weighted_mean(
        [abs(value) for value in recent_parallel], recent_weights
    )
    lateral_fraction = lateral_abs_mean / max(
        lateral_abs_mean + parallel_abs_mean, EPS
    )
    lateral_angle = math.degrees(math.atan2(lateral_abs_mean, max(parallel_abs_mean, EPS)))
    sign_coherence = abs(
        sum(weight * value for value, weight in zip(recent_perpendicular, recent_weights))
    ) / max(
        sum(weight * abs(value) for value, weight in zip(recent_perpendicular, recent_weights)),
        EPS,
    )

    centered_lateral = [
        value - perpendicular_baseline.mean for value in recent_perpendicular
    ]
    lateral_travel = abs(
        float(sum(value * dt for value, dt in zip(centered_lateral, recent_dts)))
    )
    travel_noise = perpendicular_baseline.sigma * math.sqrt(
        max(sum(float(dt) ** 2 for dt in recent_dts), EPS)
    )
    z_lateral_speed = abs(
        signed_lateral - perpendicular_baseline.mean
    ) / max(perpendicular_baseline.sigma, EPS)
    z_lateral_travel = lateral_travel / max(travel_noise, EPS)
    persistence = math.sqrt(
        max(0.0, sign_coherence * sample_reliability(
                    equivalent_temporal_sample_count(recent_count, sampling_interval_scale)
                ))
    )
    effective_lateral_z = max(z_lateral_speed, z_lateral_travel) * persistence

    lateral_confidence = combine_confidences(
        [
            detection_confidence,
            track_confidence,
            axis.confidence,
            motion_consistency,
            perpendicular_baseline.confidence,
        ]
    )
    forward_reference = abs(parallel_baseline.mean) + parallel_baseline.sigma
    forward_retention = parallel_abs_mean / max(forward_reference, EPS)
    forward_preservation = clip01(forward_retention)


    centered_lateral_abs = [abs(value) for value in centered_lateral]
    endpoint_count = min(2, len(centered_lateral_abs))
    endpoint_lateral = weighted_median(
        centered_lateral_abs[-endpoint_count:],
        recent_weights[-endpoint_count:],
    ) if endpoint_count > 0 else 0.0
    lateral_persistence = endpoint_lateral / max(
        endpoint_lateral + perpendicular_baseline.sigma, EPS
    )
    lateral_recovery = 1.0 - clip01(lateral_persistence)


    early_count = max(1, len(centered_lateral_abs) - endpoint_count)
    early_lateral = weighted_median(
        centered_lateral_abs[:early_count],
        recent_weights[:early_count],
    )
    lateral_decay_recovery = clip01(
        (early_lateral - endpoint_lateral)
        / max(
            early_lateral + perpendicular_baseline.sigma,
            EPS,
        )
    )

    side_slip_likelihood = combine_confidences(
        [
            clip01(lateral_fraction),
            1.0 - forward_preservation,
            clip01(lateral_persistence),
        ]
    )
    lane_change_likelihood = combine_confidences(
        [
            forward_preservation,
            1.0 - clip01(lateral_fraction),
            axis.confidence,
            max(context_confidence, lateral_recovery),
        ]
    )


    normal_lane_change_alternative = clip01(
        lane_change_likelihood
        * (1.0 - side_slip_likelihood)
        * (1.0 - side_slip_likelihood)
    )
    lateral_raw_score = score_from_z(effective_lateral_z, lateral_confidence)
    instantaneous_lateral_score = (
        lateral_raw_score * (1.0 - normal_lane_change_alternative)
    )


    unexplained_lateral = 1.0 - normal_lane_change_alternative
    lateral_hazard_gate = math.sqrt(
        max(
            0.0,
            sign_coherence
            * clip01(lateral_persistence)
            * clip01(unexplained_lateral),
        )
    )
    raw_lateral_z = max(z_lateral_speed, z_lateral_travel)
    signed_lateral_residual = signed_lateral - perpendicular_baseline.mean
    lateral_direction = (
        1.0
        if signed_lateral_residual > EPS
        else (-1.0 if signed_lateral_residual < -EPS else 0.0)
    )
    lateral_memory_input_z = (
        lateral_direction * raw_lateral_z * lateral_hazard_gate
    )
    lateral_observation_confidence = combine_confidences(
        [
            lateral_confidence,
            sign_coherence,
            clip01(lateral_persistence),
            clip01(unexplained_lateral),
        ]
    )
    lateral_memory_window = min(
        max(3, len(motion.vxs)),
        max(3, 2 * recent_count),
    )
    lane_change_recovery_evidence = clip01(
        lane_change_likelihood
        * lateral_decay_recovery
        * forward_preservation
    )
    lateral_history_retention = 1.0 - lane_change_recovery_evidence
    (
        temporal_lateral_z,
        lateral_memory_confidence,
        temporal_lateral_coherence,
        lateral_memory_effective_count,
        lateral_memory_count,
    ) = _update_lateral_evidence_memory(
        state,
        lateral_memory_input_z,
        lateral_observation_confidence,
        lateral_memory_window,
        history_retention=lateral_history_retention,
    )
    temporal_lateral_z *= temporal_evidence_scale
    temporal_lateral_confidence = combine_confidences(
        [
            detection_confidence,
            track_confidence,
            axis.confidence,
            perpendicular_baseline.confidence,
            lateral_memory_confidence,
        ]
    )
    persistent_lateral_score = score_from_z(
        temporal_lateral_z,
        temporal_lateral_confidence,
    )


    lateral_score = max(
        instantaneous_lateral_score,
        persistent_lateral_score,
    )


    overspeed_z = 0.0
    overspeed_ratio = 0.0
    overspeed_score = 0.0
    overspeed_confidence = 0.0
    if context_parallel:
        overspeed_z = max(
            0.0, current_parallel - parallel_baseline.mean
        ) / max(parallel_baseline.sigma, EPS)
        overspeed_ratio = current_parallel / max(
            abs(parallel_baseline.mean), parallel_baseline.sigma, EPS
        )
        longitudinal_purity = abs(current_parallel) / max(
            abs(current_parallel) + abs(current_perpendicular), EPS
        )
        overspeed_confidence = combine_confidences(
            [
                detection_confidence,
                track_confidence,
                axis.confidence,
                context_confidence,
                parallel_baseline.confidence,
            ]
        )
        overspeed_score = score_from_z(overspeed_z, overspeed_confidence) * math.sqrt(
            clip01(longitudinal_purity)
        )


    decel_score = 0.0
    decel_confidence = 0.0
    old_parallel = new_parallel = decel_z = drop_ratio = 0.0
    change_window = min(
        adaptive_recent_count(len(parallel_sequence), sampling_interval_scale),
        len(parallel_sequence) // 2,
    )
    if change_window >= 2:
        old_values = parallel_sequence[-2 * change_window : -change_window]
        new_values = parallel_sequence[-change_window:]
        old_weights = motion.confidences[-2 * change_window : -change_window]
        new_weights = motion.confidences[-change_window:]
        old_parallel = weighted_median(old_values, old_weights)
        new_parallel = weighted_median(new_values, new_weights)
        drop = max(0.0, old_parallel - new_parallel)
        old_noise = max(weighted_mad(old_values, old_weights), motion.resolution)
        new_noise = max(weighted_mad(new_values, new_weights), motion.resolution)
        change_noise = max(
            parallel_baseline.sigma,
            math.sqrt(old_noise * old_noise + new_noise * new_noise),
        )
        decel_z = drop / max(change_noise, EPS)
        drop_ratio = drop / max(abs(old_parallel), change_noise, EPS)
        prior_motion_evidence = abs(old_parallel) / max(
            abs(old_parallel) + change_noise, EPS
        )
        decel_confidence = combine_confidences(
            [
                detection_confidence,
                track_confidence,
                axis.confidence,
                motion_consistency,
                parallel_baseline.confidence,
            ]
        )
        decel_score = score_from_z(decel_z, decel_confidence) * math.sqrt(
            clip01(prior_motion_evidence)
        )


    stop_score = 0.0
    stop_confidence = 0.0
    stop_z = 0.0
    flow_stop_z = 0.0
    transition_stop_z = 0.0
    cumulative_stop_z = 0.0
    flow_stop_score = 0.0
    transition_stop_score = 0.0
    persistent_stop_score = 0.0
    persistent_flow_stop_score = 0.0
    persistent_flow_stop_raw_score = 0.0
    persistent_flow_stop_z = 0.0
    persistent_flow_stop_confidence = 0.0
    persistent_flow_stop_effective_count = 0.0
    persistent_flow_stop_count = 0
    persistent_flow_stop_window = 0
    cold_start_stop_score = 0.0
    cold_start_stop_raw_score = 0.0
    cold_start_stop_z = 0.0
    cold_start_cumulative_z = 0.0
    cold_start_stop_confidence = 0.0
    cold_start_stationary_gate = 0.0
    stable_near_stop = False
    flow_effective_z = 0.0
    flow_cumulative_z = 0.0
    flow_context_effective_count = 0.0
    flow_context_support_reliability = 0.0
    flow_context_confidence = 0.0
    flow_low_speed_persistence = 0.0
    stop_state_gate = 0.0
    flow_road_gate = 0.0
    transition_effective_z = 0.0
    temporal_stop_z = 0.0
    temporal_stop_confidence = 0.0
    stop_memory_effective_count = 0.0
    stop_memory_count = 0
    stop_memory_window = 0
    position_stability = 0.0
    staticness = 0.0
    low_speed_persistence = 0.0
    prior_motion_evidence = 0.0
    moving_reference = 0.0
    moving_reference_noise = 0.0
    moving_reference_confidence = 0.0
    flow_stop_confidence = 0.0
    transition_stop_confidence = 0.0

    recent_speed_values = [
        math.hypot(vx, vy)
        for vx, vy in zip(
            motion.vxs[-recent_count:],
            motion.vys[-recent_count:],
        )
    ]
    recent_speed_weights = motion.confidences[-recent_count:]
    recent_speed_resolutions = motion.resolutions[-recent_count:]
    recent_speed = weighted_median(recent_speed_values, recent_speed_weights)
    target_speed_dispersion = weighted_mad(
        recent_speed_values, recent_speed_weights
    )
    target_speed_noise = max(
        target_speed_dispersion,
        motion.resolution,
    )
    staticness = target_speed_noise / max(
        target_speed_noise + recent_speed, EPS
    )

    observation_count = min(len(state.observations), recent_count + 1)
    recent_observations = state.observations[-observation_count:]
    if len(recent_observations) >= 2:
        elapsed = max(
            1.0,
            float(recent_observations[-1].frame_id - recent_observations[0].frame_id),
        )
        median_dt = float(np.median(recent_dts)) if recent_dts else elapsed

        effective_scale = 1.0 / max(
            motion.resolution * max(median_dt, 1.0), EPS
        )
        displacement = math.hypot(
            recent_observations[-1].ground_x - recent_observations[0].ground_x,
            recent_observations[-1].ground_y - recent_observations[0].ground_y,
        ) / max(effective_scale, EPS)
    else:
        displacement = 0.0
        elapsed = 1.0

    expected_jitter_travel = max(
        target_speed_noise, motion.resolution
    ) * elapsed
    position_stability = 1.0 / (
        1.0 + displacement / max(expected_jitter_travel, EPS)
    )


    flow_speed = 0.0
    flow_noise = 0.0
    current_flow_speed = 0.0
    current_flow_noise = 0.0
    current_flow_confidence = 0.0
    historical_flow_support = clip01(
        historical_flow_confidence * historical_geometry_confidence
    )


    if stop_context_speeds:
        current_flow_speed = weighted_median(
            stop_context_speeds, stop_context_weights
        )
        context_resolution = weighted_median(
            stop_context_resolutions, stop_context_weights
        )
        current_flow_noise = max(
            weighted_mad(stop_context_speeds, stop_context_weights),
            context_resolution,
        )

        flow_context_effective_count = _effective_sample_size(
            stop_context_weights
        )
        flow_context_support_reliability = sample_reliability(
            int(round(flow_context_effective_count))
        )
        median_context_weight = weighted_median(
            stop_context_weights, [1.0] * len(stop_context_weights)
        )
        current_flow_confidence = combine_confidences(
            [
                stop_context_confidence,
                flow_context_support_reliability,
                median_context_weight,
            ]
        )
        flow_context_confidence = current_flow_confidence
        flow_speed = current_flow_speed
        flow_noise = current_flow_noise

        flow_stop_noise = math.sqrt(
            target_speed_noise**2 + flow_noise**2
        )
        flow_stop_z = max(0.0, flow_speed - recent_speed) / max(
            flow_stop_noise, EPS
        )

        flow_standardized_deficits: List[float] = []
        flow_low_speed_support: List[float] = []
        for speed_value, resolution in zip(
            recent_speed_values, recent_speed_resolutions
        ):
            segment_noise = math.sqrt(
                flow_noise**2
                + max(float(resolution), target_speed_dispersion, EPS) ** 2
            )
            flow_standardized_deficits.append(
                max(0.0, flow_speed - speed_value)
                / max(segment_noise, EPS)
            )
            flow_low_speed_support.append(
                clip01(
                    (flow_speed - speed_value)
                    / max(flow_speed + flow_noise, EPS)
                )
            )
        flow_cumulative_z = _weighted_standardized_sum(
            flow_standardized_deficits, recent_speed_weights
        ) * temporal_evidence_scale
        flow_low_speed_persistence = _window_weighted_mean(
            flow_low_speed_support, recent_speed_weights
        )

        stop_state_gate = (
            max(
                0.0,
                staticness
                * position_stability
                * flow_low_speed_persistence,
            )
            ** (1.0 / 3.0)
        )
        flow_road_gate = math.sqrt(clip01(flow_road_membership))
        flow_effective_z = (
            max(flow_stop_z, flow_cumulative_z)
            * stop_state_gate
            * flow_road_gate
        )
        flow_stop_confidence = combine_confidences(
            [
                detection_confidence,
                axis.confidence,
                flow_context_confidence,
                sample_reliability(
                    equivalent_temporal_sample_count(recent_count, sampling_interval_scale)
                ),
                flow_road_membership,
            ]
        )
        flow_stop_score = score_from_one_sided_z(
            flow_effective_z, flow_stop_confidence
        )


    cold_reference_speeds: List[float] = []
    cold_reference_noises: List[float] = []
    cold_reference_weights: List[float] = []
    if current_flow_confidence > 0.0 and current_flow_speed > 0.0:
        cold_reference_speeds.append(float(current_flow_speed))
        cold_reference_noises.append(float(current_flow_noise))
        cold_reference_weights.append(float(current_flow_confidence))
    if historical_flow_support > 0.0 and historical_flow_speed > 0.0:
        cold_reference_speeds.append(float(historical_flow_speed))
        cold_reference_noises.append(
            max(float(historical_flow_noise), float(motion.resolution), EPS)
        )
        cold_reference_weights.append(float(historical_flow_support))

    cold_reference_speed = 0.0
    cold_reference_noise = 0.0
    cold_reference_confidence = 0.0
    if cold_reference_speeds:
        cold_reference_speed = weighted_median(
            cold_reference_speeds, cold_reference_weights
        )
        cold_reference_noise = max(
            weighted_median(cold_reference_noises, cold_reference_weights),
            weighted_mad(cold_reference_speeds, cold_reference_weights),
            EPS,
        )
        cold_reference_confidence = clip01(
            1.0
            - (1.0 - current_flow_confidence)
            * (1.0 - historical_flow_support)
        )
        cold_start_stop_z = max(
            0.0,
            cold_reference_speed - robust_track_speed,
        ) / max(
            math.sqrt(cold_reference_noise**2 + robust_track_noise**2),
            EPS,
        )
        cold_standardized_deficits: List[float] = []
        latest_speeds = list(robust_stationarity["latest_speeds"])
        latest_resolutions = list(robust_stationarity["latest_resolutions"])
        latest_weights = list(robust_stationarity["latest_weights"])
        for robust_speed_value, robust_resolution in zip(
            latest_speeds,
            latest_resolutions,
        ):
            segment_noise = math.sqrt(
                cold_reference_noise**2
                + max(
                    float(robust_resolution),
                    robust_track_noise,
                    EPS,
                )
                ** 2
            )
            cold_standardized_deficits.append(
                max(0.0, cold_reference_speed - float(robust_speed_value))
                / max(segment_noise, EPS)
            )
        cold_start_cumulative_z = _weighted_standardized_sum(
            cold_standardized_deficits,
            latest_weights,
        ) * temporal_evidence_scale
        robust_staticness = robust_track_noise / max(
            robust_track_noise + robust_track_speed,
            EPS,
        )
        cold_start_stationary_gate = (
            max(
                0.0,
                robust_staticness
                * robust_cluster_stability
                * robust_duration_reliability,
            )
            ** (1.0 / 3.0)
        )
        cold_start_effective_z = (
            max(cold_start_stop_z, cold_start_cumulative_z)
            * math.sqrt(cold_start_stationary_gate)
            * math.sqrt(clip01(cold_start_flow_road_membership))
            * math.sqrt(clip01(cold_start_isolation_gate))
        )
        cold_start_stop_confidence = combine_confidences(
            [
                detection_confidence,
                axis.confidence,
                cold_reference_confidence,
                historical_flow_support,
                vehicle_candidate_confidence,
                sample_reliability(
                    equivalent_temporal_sample_count(len(state.observations), sampling_interval_scale)
                ),
            ]
        )
        cold_start_stop_raw_score = score_from_one_sided_z(
            cold_start_effective_z,
            cold_start_stop_confidence,
        )
        cold_start_stop_score = min(
            cold_start_stop_raw_score,
            COLD_START_STOP_SCORE_CAP,
        )


    stable_near_stop = bool(
        recent_speed <= max(target_speed_noise, EPS)
        and displacement <= max(expected_jitter_travel, EPS)
    )


    equivalent_recent_count = max(
        2,
        equivalent_temporal_sample_count(recent_count, sampling_interval_scale),
    )
    persistent_window_cap = max(
        2,
        int(
            math.ceil(
                max(24, 6 * equivalent_recent_count)
                / sampling_interval_scale
            )
        ),
    )
    persistent_flow_stop_window = min(
        max(2, int(state.seen_hits)),
        persistent_window_cap,
    )
    (
        persistent_flow_stop_z,
        persistent_flow_memory_confidence,
        persistent_flow_stop_effective_count,
        persistent_flow_stop_count,
    ) = _update_persistent_flow_stop_memory(
        state,
        flow_effective_z,
        flow_stop_confidence,
        persistent_flow_stop_window,
        stable_near_stop,
    )
    persistent_flow_stop_z *= temporal_evidence_scale
    persistent_flow_stop_confidence = combine_confidences(
        [
            detection_confidence,
            track_confidence,
            axis.confidence,
            persistent_flow_memory_confidence,
            flow_context_confidence,
            flow_road_membership,
        ]
    )
    persistent_flow_stop_raw_score = score_from_one_sided_z(
        persistent_flow_stop_z,
        persistent_flow_stop_confidence,
    )
    persistent_flow_stop_score = max(
        flow_stop_score,
        min(
            persistent_flow_stop_raw_score,
            PERSISTENT_FLOW_STOP_SCORE_CAP,
        ),
    )


    prior_speed_values = [
        math.hypot(vx, vy)
        for vx, vy in zip(
            motion.vxs[:prior_end],
            motion.vys[:prior_end],
        )
    ]
    prior_speed_weights = motion.confidences[:prior_end]
    if prior_speed_values and recent_speed_values:
        moving_reference = weighted_median(
            prior_speed_values, prior_speed_weights
        )
        moving_reference_dispersion = weighted_mad(
            prior_speed_values,
            prior_speed_weights,
            center=moving_reference,
        )
        moving_reference_noise = max(
            moving_reference_dispersion,
            historical_innovation,
            motion.resolution,
        )
        prior_effective_count = _effective_sample_size(prior_speed_weights)
        recent_effective_count = _effective_sample_size(recent_speed_weights)


        moving_reference_uncertainty = moving_reference_noise / math.sqrt(
            max(1.0, prior_effective_count)
        )
        recent_speed_uncertainty = target_speed_noise / math.sqrt(
            max(1.0, recent_effective_count)
        )
        transition_noise = math.sqrt(
            moving_reference_uncertainty**2
            + recent_speed_uncertainty**2
        )
        transition_stop_z = max(
            0.0, moving_reference - recent_speed
        ) / max(transition_noise, EPS)


        standardized_deficits: List[float] = []
        for speed_value, resolution in zip(
            recent_speed_values, recent_speed_resolutions
        ):
            segment_noise = math.sqrt(
                moving_reference_uncertainty**2
                + max(float(resolution), target_speed_dispersion, EPS) ** 2
            )
            standardized_deficits.append(
                max(0.0, moving_reference - speed_value)
                / max(segment_noise, EPS)
            )
        cumulative_stop_z = _weighted_standardized_sum(
            standardized_deficits, recent_speed_weights
        ) * temporal_evidence_scale

        low_speed_support = [
            clip01(
                (moving_reference - speed_value)
                / max(moving_reference + moving_reference_noise, EPS)
            )
            for speed_value in recent_speed_values
        ]
        low_speed_persistence = _window_weighted_mean(
            low_speed_support, recent_speed_weights
        )
        prior_motion_evidence = moving_reference / max(
            moving_reference + moving_reference_noise, EPS
        )
        moving_reference_confidence = combine_confidences(
            [
                sample_reliability(
                    equivalent_temporal_sample_count(len(prior_speed_values), sampling_interval_scale)
                ),
                motion_consistency,
                parallel_baseline.confidence,
                prior_motion_evidence,
            ]
        )
        transition_stop_confidence = combine_confidences(
            [
                detection_confidence,
                track_confidence,
                axis.confidence,
                moving_reference_confidence,
                sample_reliability(
                    equivalent_temporal_sample_count(recent_count, sampling_interval_scale)
                ),
            ]
        )


        transition_effective_z = max(
            transition_stop_z, cumulative_stop_z
        ) * math.sqrt(
            max(0.0, staticness * position_stability)
        )
        transition_stop_score = score_from_one_sided_z(
            transition_effective_z, transition_stop_confidence
        )


    instantaneous_stop_effective_z = max(flow_effective_z, transition_effective_z)
    instantaneous_stop_confidence = (
        transition_stop_confidence
        if transition_effective_z >= flow_effective_z
        else flow_stop_confidence
    )


    stop_observation_confidence = combine_confidences(
        [
            detection_confidence,
            track_confidence,
            axis.confidence,
            instantaneous_stop_confidence,
        ]
    )
    stop_memory_window = min(
        max(2, len(motion.vxs)),
        max(2, 2 * recent_count),
    )
    (
        temporal_stop_z,
        memory_only_confidence,
        stop_memory_effective_count,
        stop_memory_count,
    ) = _update_stop_evidence_memory(
        state,
        instantaneous_stop_effective_z,
        stop_observation_confidence,
        stop_memory_window,
    )
    temporal_stop_z *= temporal_evidence_scale
    temporal_stop_confidence = combine_confidences(
        [
            detection_confidence,
            track_confidence,
            axis.confidence,
            memory_only_confidence,
        ]
    )
    persistent_stop_score = score_from_one_sided_z(
        temporal_stop_z,
        temporal_stop_confidence,
    )

    stop_score = max(
        flow_stop_score,
        transition_stop_score,
        persistent_stop_score,
        persistent_flow_stop_score,
        cold_start_stop_score,
    )
    stop_z = max(
        flow_stop_z,
        transition_stop_z,
        cumulative_stop_z,
        temporal_stop_z,
        persistent_flow_stop_z,
        cold_start_stop_z,
        cold_start_cumulative_z,
    )
    if cold_start_stop_score >= max(
        flow_stop_score,
        transition_stop_score,
        persistent_stop_score,
        persistent_flow_stop_score,
    ):
        stop_confidence = cold_start_stop_confidence
    elif persistent_flow_stop_score >= max(
        flow_stop_score,
        transition_stop_score,
        persistent_stop_score,
    ):
        stop_confidence = persistent_flow_stop_confidence
    elif persistent_stop_score >= max(flow_stop_score, transition_stop_score):
        stop_confidence = temporal_stop_confidence
    elif transition_stop_score >= flow_stop_score:
        stop_confidence = combine_confidences(
            [
                detection_confidence,
                track_confidence,
                axis.confidence,
                moving_reference_confidence,
                prior_motion_evidence,
            ]
        )
    else:
        stop_confidence = flow_stop_confidence


    if static_nonvehicle_suspect:
        lateral_score = 0.0
        lateral_raw_score = 0.0
        instantaneous_lateral_score = 0.0
        persistent_lateral_score = 0.0
        overspeed_score = 0.0
        decel_score = 0.0
        flow_stop_score = 0.0
        transition_stop_score = 0.0
        persistent_stop_score = 0.0
        persistent_flow_stop_score = 0.0
        cold_start_stop_score = 0.0
        stop_score = 0.0
        state.stop_z_history.clear()
        state.stop_weight_history.clear()
        state.persistent_flow_stop_z_history.clear()
        state.persistent_flow_stop_weight_history.clear()
        state.lateral_signed_z_history.clear()
        state.lateral_weight_history.clear()
        state.impact_posterior_score = 0.0


    longitudinal_candidates = {
        "OVERSPEED": float(overspeed_score),
        "DECEL": float(decel_score),
        "STOP": float(stop_score),
    }
    strongest_longitudinal_event, strongest_longitudinal_score = max(
        longitudinal_candidates.items(),
        key=lambda item: item[1],
    )
    temporal_lateral_support = (
        clip01(temporal_lateral_coherence)
        * clip01(unexplained_lateral)
        * sample_reliability(
            equivalent_temporal_sample_count(lateral_memory_count, sampling_interval_scale)
        )
    )
    collision_lateral_support = max(
        clip01(lateral_hazard_gate),
        clip01(side_slip_likelihood * unexplained_lateral),
        clip01(temporal_lateral_support),
    )
    (
        collision_joint_energy,
        collision_balance,
        collision_gate,
        collision_joint_score,
    ) = _joint_motion_surprise(
        lateral_score,
        strongest_longitudinal_score,
        collision_lateral_support,
    )


    impact_continuation_support = clip01(
        collision_balance * collision_gate
    )
    impact_posterior_score, impact_retention = (
        _update_impact_episode_posterior(
            state,
            collision_joint_score,
            impact_continuation_support,
        )
    )
    impact_episode_score = max(
        float(collision_joint_score),
        float(impact_posterior_score),
    )


    raw_stop_score = float(stop_score)
    stop_semantic_dominant = bool(
        strongest_longitudinal_event == "STOP"
        and float(stop_score) >= float(lateral_score)
    )
    stop_semantic_score = (
        max(float(stop_score), float(impact_episode_score))
        if stop_semantic_dominant
        else float(stop_score)
    )
    impact_selection_score = (
        0.0 if stop_semantic_dominant else float(impact_episode_score)
    )

    branch_scores = {
        "LATERAL": float(lateral_score),
        "OVERSPEED": float(overspeed_score),
        "DECEL": float(decel_score),
        "STOP": float(stop_semantic_score),
        "IMPACT": float(impact_selection_score),
    }
    event, score = max(branch_scores.items(), key=lambda item: item[1])
    if score <= 0.0:
        event = "NORMAL"
    overall_confidence = combine_confidences(
        [detection_confidence, track_confidence, axis.confidence]
    )

    debug = {
        "sampling_interval_scale": float(sampling_interval_scale),
        "temporal_evidence_scale": float(temporal_evidence_scale),
        "reference_z": float(REFERENCE_Z),
        "one_sided_reference_z": float(ONE_SIDED_REFERENCE_Z),
        "overall_confidence": float(overall_confidence),
        "detection_confidence": float(detection_confidence),
        "track_confidence": float(track_confidence),
        "axis_confidence": float(axis.confidence),
        "motion_consistency": float(motion_consistency),
        "context_confidence": float(context_confidence),
        "stop_context_confidence": float(stop_context_confidence),
        "context_track_ids": [int(v) for v in used_context_track_weights],
        "context_track_weights": {
            str(k): float(v) for k, v in used_context_track_weights.items()
        },
        "stop_context_track_ids": [
            int(v) for v in used_stop_context_track_weights
        ],
        "stop_context_track_weights": {
            str(k): float(v)
            for k, v in used_stop_context_track_weights.items()
        },
        "stop_context_count": int(len(stop_context_speeds)),
        "current_flow_speed": float(current_flow_speed),
        "current_flow_noise": float(current_flow_noise),
        "current_flow_confidence": float(current_flow_confidence),
        "historical_flow_speed": float(historical_flow_speed),
        "historical_flow_noise": float(historical_flow_noise),
        "historical_flow_confidence": float(historical_flow_confidence),
        "historical_geometry_confidence": float(
            historical_geometry_confidence
        ),
        "historical_flow_support": float(historical_flow_support),
        "effective_flow_corridor_membership": float(
            effective_flow_corridor_membership
        ),
        "current_flow_road_membership": float(
            current_flow_road_membership
        ),
        "cold_start_flow_road_membership": float(
            cold_start_flow_road_membership
        ),
        "robust_track_speed": float(robust_track_speed),
        "robust_track_noise": float(robust_track_noise),
        "robust_cluster_stability": float(robust_cluster_stability),
        "robust_duration_reliability": float(robust_duration_reliability),
        "robust_motion_identity": float(robust_motion_identity),
        "stationary_peer_mass": float(stationary_peer_mass),
        "stationary_peer_count": int(stationary_peer_count),
        "cold_start_isolation_gate": float(cold_start_isolation_gate),
        "current_vehicle_identity": float(current_vehicle_identity),
        "historical_vehicle_identity": float(historical_vehicle_identity),
        "road_vehicle_identity": float(road_vehicle_identity),
        "vehicle_candidate_confidence": float(vehicle_candidate_confidence),
        "static_nonvehicle_suspect": int(static_nonvehicle_suspect),
        "display_eligible": int(not static_nonvehicle_suspect),
        "parallel_baseline_confidence": float(parallel_baseline.confidence),
        "perpendicular_baseline_confidence": float(perpendicular_baseline.confidence),
        "mu_parallel": float(parallel_baseline.mean),
        "sigma_parallel": float(parallel_baseline.sigma),
        "mu_perpendicular": float(perpendicular_baseline.mean),
        "sigma_perpendicular": float(perpendicular_baseline.sigma),
        "motion_resolution": float(motion.resolution),
        "motion_innovation": float(historical_innovation),
        "raw_recent_innovation": float(motion.innovation),
        "recent_window": int(recent_count),
        "z_lateral_speed": float(z_lateral_speed),
        "z_lateral_travel": float(z_lateral_travel),
        "effective_lateral_z": float(effective_lateral_z),
        "lateral_angle_deg": float(lateral_angle),
        "lateral_fraction": float(lateral_fraction),
        "lateral_travel": float(lateral_travel),
        "lateral_sign_coherence": float(sign_coherence),
        "forward_retention": float(forward_retention),
        "forward_preservation": float(forward_preservation),
        "lateral_endpoint": float(endpoint_lateral),
        "lateral_persistence": float(lateral_persistence),
        "lateral_recovery": float(lateral_recovery),
        "lateral_early": float(early_lateral),
        "lateral_decay_recovery": float(lateral_decay_recovery),
        "side_slip_likelihood": float(side_slip_likelihood),
        "lane_change_likelihood": float(lane_change_likelihood),
        "normal_lane_change_alternative": float(normal_lane_change_alternative),
        "lateral_raw_score": float(lateral_raw_score),
        "instantaneous_lateral_score": float(instantaneous_lateral_score),
        "persistent_lateral_score": float(persistent_lateral_score),
        "lateral_hazard_gate": float(lateral_hazard_gate),
        "lateral_memory_input_z": float(lateral_memory_input_z),
        "temporal_lateral_z": float(temporal_lateral_z),
        "temporal_lateral_confidence": float(temporal_lateral_confidence),
        "temporal_lateral_coherence": float(temporal_lateral_coherence),
        "lateral_memory_effective_count": float(lateral_memory_effective_count),
        "lateral_memory_count": int(lateral_memory_count),
        "lateral_memory_window": int(lateral_memory_window),
        "lane_change_recovery_evidence": float(lane_change_recovery_evidence),
        "lateral_history_retention": float(lateral_history_retention),
        "lateral_confidence": float(lateral_confidence),
        "overspeed_z": float(overspeed_z),
        "overspeed_ratio": float(overspeed_ratio),
        "overspeed_confidence": float(overspeed_confidence),
        "old_parallel": float(old_parallel),
        "new_parallel": float(new_parallel),
        "decel_z": float(decel_z),
        "drop_ratio": float(drop_ratio),
        "decel_confidence": float(decel_confidence),
        "recent_speed": float(recent_speed),
        "target_speed_noise": float(target_speed_noise),
        "staticness": float(staticness),
        "stop_displacement": float(displacement),
        "position_stability": float(position_stability),
        "flow_speed": float(flow_speed),
        "traffic_occupancy_confidence": float(traffic_occupancy_confidence),
        "flow_corridor_membership": float(flow_corridor_membership),
        "flow_road_membership": float(flow_road_membership),
        "stop_z": float(stop_z),
        "flow_stop_z": float(flow_stop_z),
        "flow_cumulative_z": float(flow_cumulative_z),
        "flow_context_effective_count": float(flow_context_effective_count),
        "flow_context_support_reliability": float(flow_context_support_reliability),
        "flow_context_confidence": float(flow_context_confidence),
        "flow_low_speed_persistence": float(flow_low_speed_persistence),
        "stop_state_gate": float(stop_state_gate),
        "flow_road_gate": float(flow_road_gate),
        "transition_stop_z": float(transition_stop_z),
        "cumulative_stop_z": float(cumulative_stop_z),
        "flow_stop_score": float(flow_stop_score),
        "transition_stop_score": float(transition_stop_score),
        "persistent_stop_score": float(persistent_stop_score),
        "persistent_flow_stop_score": float(persistent_flow_stop_score),
        "persistent_flow_stop_raw_score": float(
            persistent_flow_stop_raw_score
        ),
        "persistent_flow_stop_z": float(persistent_flow_stop_z),
        "persistent_flow_stop_confidence": float(
            persistent_flow_stop_confidence
        ),
        "persistent_flow_stop_effective_count": float(
            persistent_flow_stop_effective_count
        ),
        "persistent_flow_stop_count": int(persistent_flow_stop_count),
        "persistent_flow_stop_window": int(persistent_flow_stop_window),
        "cold_start_stop_score": float(cold_start_stop_score),
        "cold_start_stop_raw_score": float(cold_start_stop_raw_score),
        "cold_start_stop_z": float(cold_start_stop_z),
        "cold_start_cumulative_z": float(cold_start_cumulative_z),
        "cold_start_stop_confidence": float(cold_start_stop_confidence),
        "cold_start_stationary_gate": float(cold_start_stationary_gate),
        "cold_reference_speed": float(cold_reference_speed),
        "cold_reference_noise": float(cold_reference_noise),
        "cold_reference_confidence": float(cold_reference_confidence),
        "stable_near_stop": int(stable_near_stop),
        "flow_stop_effective_z": float(flow_effective_z),
        "transition_stop_effective_z": float(transition_effective_z),
        "temporal_stop_z": float(temporal_stop_z),
        "temporal_stop_confidence": float(temporal_stop_confidence),
        "stop_memory_effective_count": float(stop_memory_effective_count),
        "stop_memory_count": int(stop_memory_count),
        "stop_memory_window": int(stop_memory_window),
        "moving_reference": float(moving_reference),
        "moving_reference_noise": float(moving_reference_noise),
        "moving_reference_confidence": float(moving_reference_confidence),
        "prior_motion_evidence": float(prior_motion_evidence),
        "low_speed_persistence": float(low_speed_persistence),
        "stop_confidence": float(stop_confidence),
        "strongest_longitudinal_event": str(strongest_longitudinal_event),
        "strongest_longitudinal_score": float(strongest_longitudinal_score),
        "collision_lateral_support": float(collision_lateral_support),
        "collision_joint_energy": float(collision_joint_energy),
        "collision_balance": float(collision_balance),
        "collision_gate": float(collision_gate),
        "collision_joint_score": float(collision_joint_score),
        "impact_continuation_support": float(impact_continuation_support),
        "impact_retention": float(impact_retention),
        "impact_posterior_score": float(impact_posterior_score),
        "impact_episode_score": float(impact_episode_score),
        "raw_stop_score": float(raw_stop_score),
        "stop_semantic_dominant": int(stop_semantic_dominant),
        "stop_semantic_score": float(stop_semantic_score),
        "impact_selection_score": float(impact_selection_score),
    }
    return AnomalyResult(
        score=float(score),
        event=str(event),
        lateral_score=float(lateral_score),
        overspeed_score=float(overspeed_score),
        decel_score=float(decel_score),
        stop_score=float(stop_semantic_score),
        v_parallel=float(current_parallel),
        v_perpendicular=float(current_perpendicular),
        road_axis=axis,
        context_count=len(context_parallel),
        confidence=float(overall_confidence),
        debug=debug,
    )
