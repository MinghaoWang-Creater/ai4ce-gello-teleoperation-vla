import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow_datasets as tfds
from natsort import natsorted

KEY_PREFIX = "ai4ce_gello"
GELLO_OUTPUT_PATH = "./bc_data/gello"
DATASET_VALIDATION_RATIO = 0.1

IMAGE_RESOLUTION = (180, 320) # H, W
DOF = 6 # Degrees of Freedom
GRIPPER_DOF = 1 # Gripper Degrees of Freedom

BASE_IMAGE_KEY = "base_rgb"
BASE_DEPTH_KEY = "base_depth"
WRIST_IMAGE_KEY = "wrist_rgb"
WRIST_DEPTH_KEY = "wrist_depth"


"""
After setting all the constants, run the following:
tfds build gello/data_utils/gello_rlds.py
"""


def resize(image: np.ndarray, size: tuple[int, int] = IMAGE_RESOLUTION) -> np.ndarray:
    """Resize the image to the specified size."""
    image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_LINEAR) # opencv uses (W, H) for resize.
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension if grayscale
    return image

class GelloRLDSDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        """Dataset's Metadata"""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(*IMAGE_RESOLUTION, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera's RGB obervation",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(*IMAGE_RESOLUTION, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Wrist camera's RGB obervation",
                                    ),
                                    "depth_image": tfds.features.Image(
                                        shape=(*IMAGE_RESOLUTION, 1),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Main camera's depth observation",
                                    ),
                                    "wrist_depth_image": tfds.features.Image(
                                        shape=(*IMAGE_RESOLUTION, 1),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Wrist camera's depth observation",
                                    ),
                                    "joint_position": tfds.features.Tensor(
                                        shape=(DOF,),
                                        dtype=np.float64,
                                        doc="Joint position state",
                                    ),
                                    "cartesian_position": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float64,
                                        doc="Cartesian position in pose+quat format",
                                    ),
                                    "gripper_position": tfds.features.Tensor(
                                        shape=(1,),
                                        dtype=np.float64,
                                        doc="Gripper position state",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(DOF + GRIPPER_DOF,), dtype=np.float64, doc="Robot action"),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict({
                        "recording_key": tfds.features.Text(doc="Unique key for the recording."),
                        "recording_path": tfds.features.Text(doc="Path to the recording."),
                    }),
                }
            )
        )

    def _split_generators(
        self,
        dl_manager: tfds.download.DownloadManager,
    ):
        datapath = Path(GELLO_OUTPUT_PATH)
        val_ratio = DATASET_VALIDATION_RATIO

        assert datapath.exists(), f"Gello output path {datapath} does not exist."
        assert 0 <= float(val_ratio) <= 1, f"Validation ratio {val_ratio} is not valid."

        sub_dirs = natsorted(datapath.glob("*/"))
        validation_indices = set(np.random.choice(len(sub_dirs), size=int(len(sub_dirs) * float(val_ratio)), replace=False))
        train_paths = [sub_dirs[i] for i in range(len(sub_dirs)) if i not in validation_indices]
        val_paths = [sub_dirs[i] for i in range(len(sub_dirs)) if i in validation_indices]

        print("Total of {} episodes.".format(len(sub_dirs)))
        print("Total of {} training episodes.".format(len(train_paths)))
        print("Total of {} validation episodes.".format(len(val_paths)))


        # TODO: Remove this check when the dataset is ready.
        if len(val_paths) == 0:
            val_paths = train_paths[:1]

        return {
            "train": self._generate_examples(train_paths),
            "val": self._generate_examples(val_paths),
        }

    def _generate_examples(
        self,
        paths: list[Path],
    ):
        def _parse_episode(episode_path: Path):
            data_key = f"{KEY_PREFIX}_{episode_path.name}"

            pkls = natsorted(episode_path.glob("*.pkl"))[:]
            episode = {
                "steps": [],
                "episode_metadata": {
                    "recording_key": data_key,
                    "recording_path": str(episode_path),
                },
            }
            for pkl in pkls:
                try:
                    with pkl.open("rb") as f:
                        frame: dict[str, np.ndarray] = pickle.load(f)
                except Exception as e:
                    print(f"Skipping {pkl!s} due to error: {e}")
                    continue

                step = {
                    "observation": {
                        "image": resize(frame.get(BASE_IMAGE_KEY, np.empty((*IMAGE_RESOLUTION, 3), dtype=np.uint8))),
                        "wrist_image": resize(frame.get(WRIST_IMAGE_KEY, np.empty((*IMAGE_RESOLUTION, 3), dtype=np.uint8))),
                        "depth_image": resize(frame.get(BASE_DEPTH_KEY, np.empty((*IMAGE_RESOLUTION, 1), dtype=np.uint16))),
                        "wrist_depth_image": resize(frame.get(WRIST_DEPTH_KEY, np.empty((*IMAGE_RESOLUTION, 1), dtype=np.uint16))),
                        "joint_position": frame.get("joint_positions", np.empty((DOF,)))[:DOF],
                        "cartesian_position": frame.get("ee_pos_quat"),
                        "gripper_position": frame.get("gripper_position"),
                    },
                    "action": frame.get("control"),
                    "is_first": False,
                    "is_last": False,
                    "language_instruction": frame.get("language_instruction", ""),
                }
                episode["steps"].append(step)

            assert len(episode["steps"]) > 30, f"Episode {episode_path} has too few steps: {len(episode['steps'])}"
            episode["steps"][0]["is_first"] = True
            episode["steps"][-1]["is_last"] = True

            return data_key, episode

        for episode_path in paths:
            yield _parse_episode(episode_path)
