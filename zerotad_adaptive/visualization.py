from __future__ import annotations


import math
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from .common import AnomalyResult

TOP_DISPLAY_COUNT = 2


def scale_rows(rows: Sequence[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    if scale >= 0.999:
        return [dict(row) for row in rows]
    output: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "bbox_xyxy" in item:
            item["bbox_xyxy"] = tuple(float(value) * scale for value in item["bbox_xyxy"])
        for key in ("x_center", "y_center", "width", "height"):
            if key in item:
                item[key] = float(item[key]) * scale
        output.append(item)
    return output


def draw_overlay(
    frame: np.ndarray,
    observations: Sequence[Dict[str, Any]],
    results: Dict[int, AnomalyResult],
    threshold: float,
    show_axes: bool,
) -> np.ndarray:


    canvas = frame.copy()
    ranked = sorted(
        [
            row
            for row in observations
            if int(row["track_id"]) in results
            and int(
                results[int(row["track_id"])].debug.get(
                    "display_eligible",
                    1,
                )
            )
            > 0
        ],
        key=lambda row: results[int(row["track_id"])].score,
        reverse=True,
    )[:TOP_DISPLAY_COUNT]

    for rank, row in enumerate(ranked, start=1):
        track_id = int(row["track_id"])
        result = results[track_id]
        x1, y1, x2, y2 = [int(round(float(value))) for value in row["bbox_xyxy"]]
        abnormal = result.score >= threshold
        color = (0, 0, 255) if abnormal else (0, 220, 0)
        thickness = 3 if abnormal else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        status = "ALARM" if abnormal else "TOP"
        label = (
            f"#{rank} id={track_id} {status} {result.event} "
            f"S={result.score:.2f} C={result.confidence:.2f}"
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

        detail = (
            f"Lat={result.lateral_score:.2f} Over={result.overspeed_score:.2f} "
            f"Dec={result.decel_score:.2f} Stop={result.stop_score:.2f}"
        )
        cv2.putText(
            canvas,
            detail,
            (x1, min(canvas.shape[0] - 8, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            detail,
            (x1, min(canvas.shape[0] - 8, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

        if show_axes and result.road_axis.ready:
            center_x = int(round(0.5 * (x1 + x2)))
            center_y = int(round(y2))
            object_scale = math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
            length = float(np.clip(1.45 * object_scale, 18.0, 55.0))
            axis_x, axis_y = result.road_axis.x, result.road_axis.y
            perpendicular_x, perpendicular_y = -axis_y, axis_x
            cv2.arrowedLine(
                canvas,
                (center_x, center_y),
                (
                    int(round(center_x + length * axis_x)),
                    int(round(center_y + length * axis_y)),
                ),
                (255, 255, 0),
                2,
                cv2.LINE_AA,
                tipLength=0.22,
            )
            cv2.line(
                canvas,
                (
                    int(round(center_x - 0.65 * length * perpendicular_x)),
                    int(round(center_y - 0.65 * length * perpendicular_y)),
                ),
                (
                    int(round(center_x + 0.65 * length * perpendicular_x)),
                    int(round(center_y + 0.65 * length * perpendicular_y)),
                ),
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        canvas,
        "Top-2 vehicle surprise",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
