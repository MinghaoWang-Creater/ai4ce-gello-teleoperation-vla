from dataclasses import dataclass
from pathlib import Path

import tyro
import rerun as rr

from gello.visualization.visualizer import RecordingVisualizer
from gello.visualization.logger import RerunMJCFLogger, build_scene

@dataclass
class Args:
    # hostname: str = "128.32.175.167"
    recording_path: str = "./ur10e_data/gello/1008_135051"
    robot_xml_path: str = "./third_party/mujoco_menagerie/universal_robots_ur10e/ur10e.xml"
    gripper_xml_path: str = "./third_party/mujoco_menagerie/robotiq_2f85/2f85.xml"

def main(args: Args):
    rr.init("gello_visualization", spawn=True)
    arena, assets = build_scene(args.robot_xml_path, args.gripper_xml_path)
    mjcf_logger = RerunMJCFLogger(
        xml_string=arena.to_xml_string(),
        assets=assets,
    )
    vis = RecordingVisualizer()
    vis.playback(
        recording_path=Path(args.recording_path),
        mjcf_logger=mjcf_logger,
    )

if __name__ == "__main__":
    main(tyro.cli(Args))
