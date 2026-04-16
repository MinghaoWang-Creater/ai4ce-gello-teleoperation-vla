"""
Inspect a pkl file: print all fields and display camera images.
Usage:
    python inspect_pkl.py                          # latest episode, first frame
    python inspect_pkl.py <pkl_file>               # specific pkl file
    python inspect_pkl.py <episode_dir>            # first frame of that episode
    python inspect_pkl.py <episode_dir> --all      # slideshow of all frames
    python inspect_pkl.py <episode_dir> --frame 10 # specific frame index
"""
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path


def inspect_data(data: dict):
    print(f"\n{'─' * 60}")
    print(f"{'Field':<25} {'Shape/Type':<25} {'Value / Preview'}")
    print(f"{'─' * 60}")
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            shape_str = str(val.shape)
            dtype_str = str(val.dtype)
            if val.size <= 10:
                preview = np.round(val.flatten(), 4).tolist()
            else:
                preview = f"[{', '.join(f'{v:.4f}' for v in val.flatten()[:5])}, ...]"
            print(f"{key:<25} {shape_str + ' ' + dtype_str:<25} {preview}")
        elif isinstance(val, (int, float, bool)):
            print(f"{key:<25} {'scalar':<25} {round(val, 6) if isinstance(val, float) else val}")
        else:
            print(f"{key:<25} {str(type(val).__name__):<25} {val}")
    print(f"{'─' * 60}")
    print(f"Total fields: {len(data)}\n")


def show_images(data: dict, title_prefix: str = ""):
    import cv2
    image_keys = [k for k, v in data.items()
                  if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3]
    depth_keys = [k for k, v in data.items()
                  if isinstance(v, np.ndarray) and v.ndim == 2]

    if not image_keys and not depth_keys:
        print("No image data found in this frame.")
        return False

    panels = []

    for key in image_keys:
        img = data[key].copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Add label
        cv2.putText(img_bgr, key, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        panels.append(img_bgr)

    for key in depth_keys:
        depth = data[key].copy()
        # Normalize depth to 0-255 for display
        valid = depth[np.isfinite(depth)]
        if len(valid) > 0:
            d_min, d_max = valid.min(), valid.max()
            depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
        else:
            depth_norm = np.zeros_like(depth)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        cv2.putText(depth_colored, key, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(depth_colored)

    # Resize all panels to same height
    target_h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != target_h:
            scale = target_h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), target_h))
        resized.append(p)

    combined = np.concatenate(resized, axis=1)
    win_title = f"GELLO Data Inspector  {title_prefix}"
    cv2.imshow(win_title, combined)
    return True


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default=None,
                        help="pkl file or episode directory")
    parser.add_argument("--all", action="store_true",
                        help="slideshow through all frames in the episode")
    parser.add_argument("--frame", type=int, default=0,
                        help="frame index to inspect (default: 0)")
    parser.add_argument("--fps", type=int, default=10,
                        help="slideshow speed in fps (default: 10)")
    args = parser.parse_args()

    # Resolve target path
    if args.target is None:
        data_root = Path("bc_data/gello")
        eps = sorted(data_root.glob("*/"))
        if not eps:
            print("No episodes found in bc_data/gello/")
            sys.exit(1)
        target = eps[-1]
        print(f"Using latest episode: {target.name}")
    else:
        target = Path(args.target)

    if target.is_dir():
        pkls = sorted(target.glob("*.pkl"))
        if not pkls:
            print(f"No pkl files in {target}")
            sys.exit(1)
        ep_name = target.name
    elif target.suffix == ".pkl":
        pkls = [target]
        ep_name = target.stem
    else:
        print(f"Unknown target: {target}")
        sys.exit(1)

    print(f"Episode: {ep_name}  |  Total frames: {len(pkls)}")

    if args.all:
        # Slideshow mode
        import cv2
        delay = max(1, int(1000 / args.fps))
        print(f"Slideshow at {args.fps} fps. Press Q to quit, SPACE to pause.")
        paused = False
        i = 0
        while i < len(pkls):
            data = load_pkl(pkls[i])
            has_img = show_images(data, title_prefix=f"[{i+1}/{len(pkls)}] {pkls[i].name}")
            if not has_img:
                break
            key = cv2.waitKey(1 if paused else delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            if not paused:
                i += 1
        cv2.destroyAllWindows()
    else:
        # Single frame mode
        idx = min(args.frame, len(pkls) - 1)
        pkl_path = pkls[idx]
        print(f"Frame {idx}: {pkl_path.name}")
        data = load_pkl(pkl_path)

        inspect_data(data)

        import cv2
        has_img = show_images(data, title_prefix=f"frame {idx}")
        if has_img:
            print("Press any key to close the image window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
