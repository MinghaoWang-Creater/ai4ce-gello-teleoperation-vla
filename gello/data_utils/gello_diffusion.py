import pickle
from pathlib import Path

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from natsort import natsorted
from tqdm import tqdm

KEY_PREFIX = "ai4ce_gello"
GELLO_OUTPUT_PATH = "./bc_data/pick_cube" \
""

IMAGE_RESOLUTION = (180, 320)  # H, W
DOF = 6  # Degrees of Freedom
GRIPPER_DOF = 1  # Gripper Degrees of Freedom

BASE_IMAGE_KEY = "base_rgb"
BASE_DEPTH_KEY = "base_depth"
WRIST_IMAGE_KEY = "wrist_rgb"
WRIST_DEPTH_KEY = "wrist_depth"


def resize(image: np.ndarray, size: tuple[int, int] = IMAGE_RESOLUTION) -> np.ndarray:
    """Resize the image to the specified size and convert to (C, H, W) format."""
    image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_LINEAR)  # opencv uses (W, H) for resize.
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension if grayscale
    # Convert from (H, W, C) to (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    return image


def convert_to_zarr_format(output_path: str = "./bc_data/pick_cube.zarr"):
    """
    Convert the existing pickle dataset to zarr format for ReplayBuffer.

    Args:
        output_path: Path where the zarr dataset will be saved
    """
    print("Starting conversion to zarr format...")

    # Initialize data path
    datapath = Path(GELLO_OUTPUT_PATH)
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    
    if not datapath.exists():
        raise FileNotFoundError(f"Gello output path {datapath} does not exist.")
    
    # Get all episode directories
    sub_dirs = natsorted(datapath.glob("*/"))
    print(f"Found {len(sub_dirs)} episodes to convert")
    
    if len(sub_dirs) == 0:
        raise ValueError("No episodes found in the dataset directory")
    
    # First pass: Calculate total steps and collect episode info
    print("Analyzing dataset structure...")
    episode_lengths = []
    total_steps = 0
    
    for episode_path in tqdm(sub_dirs, desc="Analyzing episodes"):
        pkls = natsorted(episode_path.glob("*.pkl"))
        episode_length = 0
        
        for pkl in pkls:
            try:
                with pkl.open("rb") as f:
                    frame = pickle.load(f)
                    # Validate that required keys exist
                    if all(key in frame for key in [BASE_IMAGE_KEY, "joint_positions", "control"]):
                        episode_length += 1
            except Exception as e:
                print(f"Warning: Skipping {pkl} due to error: {e}")
                continue
        
        if episode_length > 30:  # Filter episodes with too few steps
            episode_lengths.append(episode_length)
            total_steps += episode_length
        else:
            print(f"Warning: Skipping episode {episode_path.name} with only {episode_length} steps")
    
    print(f"Total valid episodes: {len(episode_lengths)}")
    print(f"Total steps: {total_steps}")
    
    if total_steps == 0:
        raise ValueError("No valid data found in the dataset")
    
    # Calculate episode ends (cumulative)
    episode_ends = np.cumsum(episode_lengths)
    print(f"Episode ends: {episode_ends[:5]}..." if len(episode_ends) > 5 else f"Episode ends: {episode_ends}")
    
    # Determine state and action dimensions
    state_dim = DOF + 7 + GRIPPER_DOF  # joint_position + cartesian_position + gripper_position
    action_dim = DOF + GRIPPER_DOF
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create zarr arrays with optimal chunking
    chunk_size = min(1000, total_steps // 10) if total_steps > 1000 else total_steps
    
    # Create zarr store using zarr 2.x format
    root = zarr.open_group(str(output_path_obj), mode='w')
    
    # Create data group
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # Create datasets with compression using zarr 2.x syntax
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    
    # Image data: (total_steps, C, H, W)
    img_array = data_group.create_dataset(
        'img',
        shape=(total_steps, 3, *IMAGE_RESOLUTION),
        dtype=np.uint8,
        chunks=(chunk_size, 3, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1]),
        compressor=compressor
    )
    
    # State data: (total_steps, state_dim)
    state_array = data_group.create_dataset(
        'state',
        shape=(total_steps, state_dim),
        dtype=np.float32,
        chunks=(chunk_size, state_dim),
        compressor=compressor
    )
    
    # Action data: (total_steps, action_dim)
    action_array = data_group.create_dataset(
        'action',
        shape=(total_steps, action_dim),
        dtype=np.float32,
        chunks=(chunk_size, action_dim),
        compressor=compressor
    )
    
    # Episode ends: (n_episodes,)
    episode_ends_array = meta_group.create_dataset(
        'episode_ends',
        data=episode_ends.astype(np.int64),
        compressor=compressor
    )
    
    print("Created zarr arrays, now filling with data...")
    
    # Second pass: Fill the arrays
    current_step = 0
    valid_episode_idx = 0
    
    for episode_path in tqdm(sub_dirs, desc="Converting episodes"):
        pkls = natsorted(episode_path.glob("*.pkl"))
        episode_data = []
        
        # Collect episode data
        for pkl in pkls:
            try:
                with pkl.open("rb") as f:
                    frame = pickle.load(f)
                    
                # Validate required keys
                if not all(key in frame for key in [BASE_IMAGE_KEY, "joint_positions", "control"]):
                    continue
                    
                episode_data.append(frame)
                    
            except Exception as e:
                print(f"Warning: Skipping {pkl} due to error: {e}")
                continue
        
        # Skip episodes that are too short
        if len(episode_data) <= 30:
            continue
        
        # Process each step in the episode
        for step_idx, frame in enumerate(episode_data):
            try:
                # Process image
                image = resize(frame.get(BASE_IMAGE_KEY, np.zeros((*IMAGE_RESOLUTION, 3), dtype=np.uint8)))
                img_array[current_step] = image
                
                # Process state (concatenate joint_position + cartesian_position + gripper_position)
                joint_pos = frame.get("joint_positions", np.zeros(DOF))[:DOF]
                cartesian_pos = frame.get("ee_pos_quat", np.zeros(7))
                gripper_pos = frame.get("gripper_position", np.zeros(1))
                
                # Ensure proper shapes and types
                joint_pos = np.array(joint_pos, dtype=np.float32).reshape(-1)[:DOF]
                cartesian_pos = np.array(cartesian_pos, dtype=np.float32).reshape(-1)[:7]
                gripper_pos = np.array(gripper_pos, dtype=np.float32).reshape(-1)[:1]
                
                # Pad if necessary
                if len(joint_pos) < DOF:
                    joint_pos = np.pad(joint_pos, (0, DOF - len(joint_pos)))
                if len(cartesian_pos) < 7:
                    cartesian_pos = np.pad(cartesian_pos, (0, 7 - len(cartesian_pos)))
                if len(gripper_pos) < 1:
                    gripper_pos = np.pad(gripper_pos, (0, 1 - len(gripper_pos)))
                
                state = np.concatenate([joint_pos, cartesian_pos, gripper_pos])
                state_array[current_step] = state
                
                # Process action
                action = frame.get("control", np.zeros(action_dim))
                action = np.array(action, dtype=np.float32).reshape(-1)[:action_dim]
                if len(action) < action_dim:
                    action = np.pad(action, (0, action_dim - len(action)))
                action_array[current_step] = action
                
                current_step += 1
                
            except Exception as e:
                print(f"Warning: Error processing step {step_idx} in episode {episode_path.name}: {e}")
                continue
        
        valid_episode_idx += 1
    
    # Save metadata
    root.attrs['total_steps'] = total_steps
    root.attrs['num_episodes'] = len(episode_lengths)
    root.attrs['state_dim'] = state_dim
    root.attrs['action_dim'] = action_dim
    root.attrs['image_shape'] = (3,) + IMAGE_RESOLUTION  # (C, H, W)
    root.attrs['dataset_version'] = '1.0.0'
    
    print("\nConversion completed successfully!")
    print(f"Zarr dataset saved to: {output_path_obj}")
    print(f"Total steps processed: {current_step}")
    print(f"Total episodes: {len(episode_lengths)}")
    print("Dataset structure:")
    print(f"  - data/img: {img_array.shape} ({img_array.dtype})")
    print(f"  - data/state: {state_array.shape} ({state_array.dtype})")
    print(f"  - data/action: {action_array.shape} ({action_array.dtype})")
    print(f"  - meta/episode_ends: {episode_ends_array.shape} ({episode_ends_array.dtype})")
    
    return output_path_obj


def main():
    """Main function to convert the dataset to zarr format."""
    try:
        output_path = convert_to_zarr_format()
        
        # Verify the created dataset
        print(f"\nVerifying created dataset...")
        root = zarr.open_group(str(output_path), mode='r')
        
        print("Dataset verification:")
        print(f"  Root attributes: {dict(root.attrs)}")
        print(f"  Data groups: {list(root['data'].keys())}")
        print(f"  Meta groups: {list(root['meta'].keys())}")
        
        # Check episode ends consistency
        episode_ends = root['meta']['episode_ends'][:]
        total_steps = root.attrs['total_steps']
        
        if len(episode_ends) > 0 and episode_ends[-1] == total_steps:
            print("Episode ends are consistent with total steps")
        else:
            print(f"Warning: Episode ends inconsistency. Last episode end: {episode_ends[-1] if len(episode_ends) > 0 else 'None'}, Total steps: {total_steps}")
        
        print(f"\nDataset conversion completed successfully!")
        print(f"You can now use this zarr dataset with ReplayBuffer.")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
