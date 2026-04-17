"""
Evaluate a trained Diffusion Policy checkpoint in MuJoCo simulation.

Runs offscreen (no display needed). Saves a side-by-side MP4 video
(base_cam | wrist_cam) for each episode to --output directory.

Usage (on server):
    /venv/robodiff/bin/python experiments/eval_policy.py \
        --ckpt /path/to/checkpoint.ckpt \
        --output eval_videos/ \
        --episodes 5 \
        --device cuda:0
"""

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

# Must be set before mujoco is imported — selects EGL for headless offscreen rendering
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import torch

# Make project root and diffusion_policy importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "diffusion_policy"))

from gello.robots.sim_robot_grasp import build_scene_with_cube, CUBE_CENTER, CUBE_RANGE


# ──────────────────────────────────────────────────────────────
# Scene / model helpers
# ──────────────────────────────────────────────────────────────

def _patch_xml(xml_path: str) -> str:
    """Return a path to an XML with legacy-incompatible options replaced.

    Older dm_control does not recognise integrator='implicitfast' — replace
    with 'implicit' which is functionally similar and universally supported.
    Returns the original path if no patching is needed.
    """
    with open(xml_path, "r") as f:
        content = f.read()
    if "implicitfast" not in content:
        return xml_path
    content = content.replace("implicitfast", "implicit")
    # Must write to same directory as original so relative asset paths resolve correctly
    original_dir = os.path.dirname(os.path.abspath(xml_path))
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", delete=False, mode="w", dir=original_dir
    )
    tmp.write(content)
    tmp.close()
    print(f"[patch] replaced 'implicitfast' → 'implicit' in {xml_path}")
    return tmp.name


def build_mujoco_model(robot_xml: str, gripper_xml: str):
    robot_xml  = _patch_xml(robot_xml)
    gripper_xml = _patch_xml(gripper_xml)
    arena = build_scene_with_cube(robot_xml, gripper_xml)
    assets = {
        asset.file.get_vfs_filename(): asset.file.contents
        for asset in arena.asset.all_children()
        if asset.tag == "mesh"
    }
    xml_string = arena.to_xml_string()

    # Remove keyframes: after scene composition the qpos size changes (cube freejoint
    # adds 7 dims), so any keyframe from the original robot XML becomes invalid.
    root_elem = ET.fromstring(xml_string)
    for parent in root_elem.iter():
        for child in list(parent):
            if child.tag == "keyframe":
                parent.remove(child)
    xml_string = ET.tostring(root_elem, encoding="unicode")

    model = mujoco.MjModel.from_xml_string(xml_string, assets)
    data = mujoco.MjData(model)
    return model, data


RESET_JOINTS = np.array([1.4901, -1.4725, 1.8906, -1.7993, -1.6091, 0.0196])


def reset_episode(model, data, cube_center, cube_range, cube_size=0.025):
    mujoco.mj_resetData(model, data)

    # Set arm to the same initial pose used during data collection so that the
    # policy receives in-distribution observations from the very first step.
    data.qpos[:6] = RESET_JOINTS
    data.ctrl[:6] = RESET_JOINTS   # position actuators: hold the pose

    # Randomize cube position
    lo = cube_center - cube_range
    hi = cube_center + cube_range
    xy = np.random.uniform(lo, hi)
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    adr = int(model.jnt_qposadr[cube_jid])
    data.qpos[adr:adr + 3] = [xy[0], xy[1], cube_size]
    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


# ──────────────────────────────────────────────────────────────
# Observation helpers
# ──────────────────────────────────────────────────────────────

IMG_H, IMG_W = 180, 320
N_OBS_STEPS = 2


_RENDER_WARNED = set()

def render_camera(renderer, data, camera_name: str) -> np.ndarray:
    try:
        renderer.update_scene(data, camera=camera_name)
        img = renderer.render().copy()
        return img
    except Exception as e:
        if camera_name not in _RENDER_WARNED:
            print(f"[WARN] render_camera '{camera_name}' failed: {e}")
            _RENDER_WARNED.add(camera_name)
        return np.zeros((480, 640, 3), dtype=np.uint8)


# Find site_id once at startup to mirror training's sim_robot_grasp.py behaviour.
# Training used mj_name2id(model, 6, "attachment_site") which returns -1 after
# dm_control namespacing (actual name is "ur5e/attachment_site"), so it fell back
# to site_xpos[-1].  We replicate that exactly.
def _find_ee_site_id(model) -> int:
    for name in ("attachment_site", "ur5e/attachment_site"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            print(f"[info] ee site found: '{name}' id={sid}")
            return sid
    # Fallback: same as training (mj_name2id returned -1, Python used xpos[-1])
    print("[info] ee site not found by name, using site_xpos[-1] (mirrors training)")
    return -1


def get_obs(model, data, renderer, ee_site_id: int) -> dict:
    base_raw   = render_camera(renderer, data, "base_cam")
    wrist_raw  = render_camera(renderer, data, "ur5e/wrist_cam")
    base_img   = cv2.resize(base_raw,  (IMG_W, IMG_H))   # (H, W, 3)
    wrist_img  = cv2.resize(wrist_raw, (IMG_W, IMG_H))

    joint_positions = data.qpos[:6].copy()

    # Mirror training: use site_xpos[ee_site_id] where -1 means last site
    ee_pos  = data.site_xpos[ee_site_id].copy()
    ee_mat  = data.site_xmat[ee_site_id].copy()
    ee_quat = np.zeros(4)
    mujoco.mju_mat2Quat(ee_quat, ee_mat)

    gripper_pos = data.qpos[6:7].copy()   # right_driver_joint
    state = np.concatenate([joint_positions, ee_pos, ee_quat, gripper_pos])  # (14,)

    return {
        "base_img":  base_img,
        "wrist_img": wrist_img,
        "state":     state.astype(np.float32),
        "base_raw":  base_raw,    # full-res for video recording
        "wrist_raw": wrist_raw,
    }


def build_obs_dict(history: deque, device: torch.device) -> dict:
    base_imgs  = np.stack([h["base_img"]  for h in history])   # (T, H, W, 3)
    wrist_imgs = np.stack([h["wrist_img"] for h in history])
    states     = np.stack([h["state"]     for h in history])   # (T, 14)

    # (T, H, W, 3) → (T, 3, H, W), normalize to [0, 1]
    base_t  = torch.from_numpy(np.moveaxis(base_imgs,  -1, 1).astype(np.float32) / 255.0)
    wrist_t = torch.from_numpy(np.moveaxis(wrist_imgs, -1, 1).astype(np.float32) / 255.0)
    state_t = torch.from_numpy(states)

    return {
        "base_img":  base_t.unsqueeze(0).to(device),    # (1, T, 3, H, W)
        "wrist_img": wrist_t.unsqueeze(0).to(device),
        "state":     state_t.unsqueeze(0).to(device),   # (1, T, 14)
    }


# ──────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: torch.device):
    import dill
    import hydra

    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="/tmp/eval_output")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.to(device)
    print(f"Policy loaded from {ckpt_path}")
    print(f"  n_obs_steps:    {policy.n_obs_steps}")
    print(f"  n_action_steps: {policy.n_action_steps}")
    return policy


# ──────────────────────────────────────────────────────────────
# Video writer
# ──────────────────────────────────────────────────────────────

def make_video_writer(output_path: str, fps: int = 30, width: int = 1280, height: int = 480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def write_frame(writer, base_raw: np.ndarray, wrist_raw: np.ndarray):
    frame = np.hstack([base_raw, wrist_raw])   # (480, 1280, 3) RGB
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# ──────────────────────────────────────────────────────────────
# Main eval loop
# ──────────────────────────────────────────────────────────────

def run_episode(model, data, renderer, ee_site_id, policy, device, max_steps: int, writer):
    n_obs_steps    = policy.n_obs_steps
    n_action_steps = policy.n_action_steps

    obs_history  = deque(maxlen=n_obs_steps)
    action_queue = deque()

    # Warm-up: step the physics a few times at reset pose so the arm settles,
    # then fill obs history with distinct frames instead of repeating one frame.
    for _ in range(10):
        data.ctrl[:6] = RESET_JOINTS
        mujoco.mj_step(model, data)
    for _ in range(n_obs_steps):
        obs_history.append(get_obs(model, data, renderer, ee_site_id))
        mujoco.mj_step(model, data)

    for step in range(max_steps):
        if len(action_queue) == 0:
            obs_dict = build_obs_dict(obs_history, device)
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
            actions = result["action"][0].cpu().numpy()   # (n_action_steps, 7)
            action_queue.extend(actions)

        action = action_queue.popleft()   # (7,)
        data.ctrl[:6] = action[:6]        # arm joints (rad)
        data.ctrl[6]  = float(action[6]) * 255.0   # gripper: [0,1] → [0,255]
        mujoco.mj_step(model, data)

        obs = get_obs(model, data, renderer, ee_site_id)
        obs_history.append(obs)

        if writer is not None:
            write_frame(writer, obs["base_raw"], obs["wrist_raw"])

        if step % 50 == 0:
            print(f"  step {step:4d} | gripper_cmd={action[6]:.3f} | ctrl[6]={data.ctrl[6]:.1f}")

    return step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True,  help=".ckpt 文件路径")
    parser.add_argument("--output",      default="eval_videos", help="输出目录")
    parser.add_argument("--episodes",    type=int, default=5,   help="评估 episode 数")
    parser.add_argument("--max_steps",   type=int, default=300, help="每 episode 最大步数")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--robot_xml",   default="./third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    parser.add_argument("--gripper_xml", default="./third_party/mujoco_menagerie/robotiq_2f85/2f85.xml")
    parser.add_argument("--fps",         type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_xml   = Path(args.robot_xml).resolve()
    gripper_xml = Path(args.gripper_xml).resolve()
    print(f"Building MuJoCo model from {robot_xml}")

    model, data = build_mujoco_model(str(robot_xml), str(gripper_xml))
    renderer = mujoco.Renderer(model, height=480, width=640)
    ee_site_id = _find_ee_site_id(model)

    # Find cube freejoint qpos address for reset
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_adr = int(model.jnt_qposadr[cube_jid])
    print(f"MuJoCo model: {model.nu} actuators, cube qpos_adr={cube_qpos_adr}")

    policy = load_policy(args.ckpt, device)

    for ep in range(args.episodes):
        print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
        reset_episode(model, data, CUBE_CENTER, CUBE_RANGE)

        video_path = str(output_dir / f"episode_{ep:03d}.mp4")
        writer = make_video_writer(video_path, fps=args.fps)

        steps = run_episode(model, data, renderer, ee_site_id, policy, device, args.max_steps, writer)

        writer.release()
        print(f"  Saved {steps} frames → {video_path}")

    print(f"\nDone. Videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
