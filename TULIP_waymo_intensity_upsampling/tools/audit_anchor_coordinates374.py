"""Report saved anchor poses and a depth-dependent projection example."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path(
        "/mnt/w/its_outputs/priority_validation374_abc_gXMLfD/A/db.parquet"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    for row in pd.read_parquet(args.db).itertuples():
        meta = json.loads(Path(row.anchor_png).with_name("anchor.json").read_text())
        vehicle = np.array(meta["anchor_pose_T_wv"]).reshape(4, 4)
        world_to_camera = np.array(meta["world_to_cam"]).reshape(4, 4)
        camera_in_vehicle = np.linalg.inv(world_to_camera @ vehicle)
        camera_world = np.linalg.inv(world_to_camera)
        rotation = vehicle[:3, :3]
        rows.append(dict(segment=row.db_seg, anchor_id=row.anchor_id,
                         x=vehicle[0, 3], y=vehicle[1, 3], z=vehicle[2, 3],
                         yaw_deg=np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])),
                         camera_x_world=camera_world[0, 3], camera_y_world=camera_world[1, 3],
                         camera_z_world=camera_world[2, 3],
                         camera_x_vehicle=camera_in_vehicle[0, 3],
                         camera_y_vehicle=camera_in_vehicle[1, 3],
                         camera_z_vehicle=camera_in_vehicle[2, 3],
                         image=row.anchor_png))
        if not examples:
            fx, fy, cx, cy = meta["intrinsic"]
            for distance in (5., 10., 30., 60.):
                # Hypothetical points on the vehicle forward axis, not LiDAR observations.
                camera = world_to_camera @ vehicle @ np.array([distance, 0., 0., 1.])
                depth = camera[0]
                if depth <= 0:
                    raise ValueError("Forward-axis example is behind camera")
                u, v = cx - fx * camera[1] / depth, cy - fy * camera[2] / depth
                examples.append(dict(vehicle_forward_distance_m=distance, u=u, v=v,
                                     camera_depth_m=depth,
                                     inside_image=0 <= u < meta["width"] and 0 <= v < meta["height"]))
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "anchors.csv", index=False)
    gaps = []
    for segment, group in table.groupby("segment", sort=False):
        group = group.sort_values("anchor_id", key=lambda x: x.astype(int))
        xy = group[["x", "y"]].to_numpy()
        spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        gaps.append(dict(segment=segment, anchors=len(group),
                         adjacent_xy_min_m=float(spacing.min()) if len(spacing) else None,
                         adjacent_xy_median_m=float(np.median(spacing)) if len(spacing) else None,
                         adjacent_xy_max_m=float(spacing.max()) if len(spacing) else None))
    pd.DataFrame(gaps).to_csv(args.output / "spacing.csv", index=False)
    pd.DataFrame(examples).to_csv(args.output / "forward_axis_example.csv", index=False)
    print(pd.DataFrame(gaps).to_string(index=False))
    print("Example: hypothetical vehicle-forward points for the first anchor")
    print(pd.DataFrame(examples).to_string(index=False))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
