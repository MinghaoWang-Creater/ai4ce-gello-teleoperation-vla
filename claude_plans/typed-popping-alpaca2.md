# Fix: base_cam offscreen 渲染全黑

## Context

`mujoco.Renderer` 与 `mujoco.viewer.launch_passive` 在同一进程中争用 OpenGL context，导致 Renderer 渲染结果全黑。在 `import mujoco` 之前强制将 Renderer 后端设为 EGL 即可解决。

## 修改

`gello/robots/sim_robot_grasp.py` 顶部，在 `import mujoco` 之前插入：
```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")
```

---

# Plan: 新增含立方体抓取场景的 Mujoco 仿真环境

## Context

用户已成功用 gello 遥操作 Mujoco 中的 UR5 机械臂。现在希望在仿真中加入一个可自由运动的立方体，用于：
1. 遥操作机械臂抓取立方体（采集 demo）
2. 将 demo 保存为 diffusion policy 所需的数据集格式

**要求：不改动现有的遥操作代码**，新建独立的仿真环境。

---

## 设计思路

| 层级 | 现有（不改动） | 新建 |
|------|--------------|------|
| 仿真服务器 | `MujocoRobotServer` in `sim_robot.py` | `MujocoGraspServer` in `sim_robot_grasp.py` |
| 启动入口 | `launch_nodes.py --robot sim_ur` | 在 `launch_nodes.py` 新增 `--robot sim_ur_cube` 选项（仅 4 行） |
| 控制 + 采集 | `run_env.py --agent=gello` | **完全复用，不改动** |
| ZMQ 通信 | `ZMQRobotServer` / `ZMQClientRobot` | **完全复用，不改动** |

---

## 关键文件

- **新建**：`gello/robots/sim_robot_grasp.py`
- **微改**：`experiments/launch_nodes.py`（在 `launch_robot_server` 函数中追加 `elif args.robot == "sim_ur_cube":` 分支，约 8 行）
- 不改动：`gello/robots/sim_robot.py`、`gello/env.py`、`experiments/run_env.py`

---

## 新文件：`gello/robots/sim_robot_grasp.py`

### 场景构建函数 `build_scene_with_cube()`

在现有 `build_scene()` 基础上扩展：
1. 复用 `sim_robot.py` 中的 `attach_hand_to_arm()` 函数（直接 import）
2. 向 `arena.worldbody` 添加地板 geom（`type="plane"`）
3. 将机械臂通过 `arena.worldbody.attach(arm_mjcf)` 挂载（与原来相同）
4. 向 `arena.worldbody.add("body", name="cube", ...)` 添加立方体：
   - `freejoint`：使立方体可在物理中自由运动（6 DOF）
   - `geom type="box"`：大小 5cm×5cm×5cm（`size=[0.025, 0.025, 0.025]`），红色
   - 初始位置：`(0.4, 0.0, 0.025)`——在机械臂正前方、地板上方
   - 质量：0.1 kg，摩擦系数：0.8

```python
def build_scene_with_cube(robot_xml_path, gripper_xml_path=None,
                          cube_size=0.025, cube_init_pos=(0.4, 0.0, 0.025),
                          cube_rgba=(1.0, 0.0, 0.0, 1.0)):
    arena = mjcf.RootElement()
    # 地板
    arena.worldbody.add("geom", name="floor", type="plane",
                        size=[2, 2, 0.1], pos=[0, 0, 0],
                        rgba=[0.8, 0.8, 0.8, 1.0],
                        friction=[0.8, 0.005, 0.0001])
    # 机械臂
    arm_mjcf = mjcf.from_path(str(robot_xml_path))
    if gripper_xml_path is not None:
        gripper_mjcf = mjcf.from_path(str(gripper_xml_path))
        attach_hand_to_arm(arm_mjcf, gripper_mjcf)
    arena.worldbody.attach(arm_mjcf)
    # 立方体（自由体）
    cube_body = arena.worldbody.add("body", name="cube", pos=list(cube_init_pos))
    cube_body.add("freejoint", name="cube_joint")
    cube_body.add("geom", type="box",
                  size=[cube_size]*3, rgba=list(cube_rgba),
                  contype=1, conaffinity=1, mass=0.1)
    return arena
```

### `MujocoGraspServer` 类

与 `MujocoRobotServer` 结构完全一致，差异：

1. `__init__`：调用 `build_scene_with_cube()` 替代 `build_scene()`；调试 XML 保存为 `arena_grasp.xml`（不覆盖原文件）
2. `get_observations()`：在原有 4 个字段基础上额外返回：
   ```python
   cube_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "cube")
   obs["cube_pos"]  = self._data.xpos[cube_id].copy()   # shape (3,)
   obs["cube_quat"] = self._data.xquat[cube_id].copy()  # shape (4,) [w, x, y, z]
   ```
3. 其余方法（`num_dofs`、`command_joint_state`、`serve`、`stop`、`freedrive_enabled`）与原类完全相同

完整 observations 字段：

| 字段 | 含义 | Shape |
|------|------|-------|
| `joint_positions` | 关节角度 | (7,) arm+gripper |
| `joint_velocities` | 关节速度 | (7,) |
| `ee_pos_quat` | 末端执行器位姿 | (7,) [x,y,z,qw,qx,qy,qz] |
| `gripper_position` | 夹爪开合量 | scalar |
| `cube_pos` | 立方体世界坐标 | (3,) |
| `cube_quat` | 立方体姿态四元数 | (4,) [w,x,y,z] |

---

## 微改：`experiments/launch_nodes.py`

在 `launch_robot_server()` 函数的 `elif args.robot == "sim_panda":` 之前插入：

```python
elif args.robot == "sim_ur_cube":
    from gello.robots.sim_robot_grasp import MujocoGraspServer
    MENAGERIE_ROOT = Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
    xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
    gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
    server = MujocoGraspServer(xml_path=xml, gripper_xml_path=gripper_xml,
                               port=port, host=args.hostname)
    server.serve()
```

---

## 使用方式（采集数据）

**终端 1**：启动含立方体的仿真环境
```bash
python experiments/launch_nodes.py --robot sim_ur_cube
```

**终端 2**：启动遥操作 + 数据采集
```bash
python experiments/run_env.py --agent=gello --use_save_interface
```
键盘操作：
- `S`：开始记录当前 episode
- `C`（每帧自动）：保存帧
- `Q`：停止记录
- `Ctrl+C`：退出

保存路径：`./bc_data/gello/MMDD_HHMMSS/*.pkl`  
每个 `.pkl` 包含完整 observations（含 `cube_pos`、`cube_quat`）+ `control` 动作。

---

---

## 相机设计（仿真 offscreen 渲染）

### 为何需要在仿真中加相机

`gello_diffusion.py` 要求每帧 pickle 中含 `"base_rgb"` 字段；如不渲染相机图像，仿真数据将无法用于 diffusion policy 训练。

### 相机添加方式

在 `build_scene_with_cube()` 中：

**全局基础相机**（固定于世界坐标系）
```python
arena.worldbody.add(
    "camera", name="base_cam",
    pos=[1.2, 0.0, 1.0],
    xyaxes=[0, -1, 0, -0.6, 0, 0.8],  # 侧前方俯视工作台
)
```

**腕部相机**（固定于夹爪 base body）
```python
# 在 arm_mjcf 附加到 arena 之前，找到 wrist_3_link 加入相机
wrist_body = arm_mjcf.find("body", "wrist_3_link")
wrist_body.add(
    "camera", name="wrist_cam",
    pos=[0.0, 0.0, -0.08],  # 在腕部关节下方
    euler=[0, 180, 0],       # 朝向末端（向下看）
)
```

> 附加到 arena 后，腕部相机的 MuJoCo 名称变为 `"ur5e/wrist_cam"`（dm_control 自动加 model name 前缀）。基础相机名仍为 `"base_cam"`。

### offscreen 渲染（在 `MujocoGraspServer` 中）

```python
# __init__ 中创建渲染器（非线程安全，需在物理循环线程中使用）
self._renderer = mujoco.Renderer(self._model, height=480, width=640)
self._depth_renderer = mujoco.Renderer(self._model, height=480, width=640)
self._depth_renderer.enable_depth_rendering()

# 渲染结果缓存
self._base_rgb   = np.zeros((480, 640, 3), dtype=np.uint8)
self._base_depth = np.zeros((480, 640), dtype=np.float32)
self._wrist_rgb  = np.zeros((480, 640, 3), dtype=np.uint8)
self._wrist_depth= np.zeros((480, 640), dtype=np.float32)
```

在 `serve()` 物理循环中（`mj_step` 之后）：
```python
# 渲染基础相机 RGB
self._renderer.update_scene(self._data, camera="base_cam")
self._base_rgb = self._renderer.render().copy()

# 渲染基础相机 depth
self._depth_renderer.update_scene(self._data, camera="base_cam")
self._base_depth = self._depth_renderer.render().copy()

# 渲染腕部相机 RGB（相机名含命名空间前缀）
self._renderer.update_scene(self._data, camera="ur5e/wrist_cam")
self._wrist_rgb = self._renderer.render().copy()

self._depth_renderer.update_scene(self._data, camera="ur5e/wrist_cam")
self._wrist_depth = self._depth_renderer.render().copy()
```

`get_observations()` 补充字段：
```python
obs["base_rgb"]    = self._base_rgb     # (480, 640, 3) uint8
obs["base_depth"]  = self._base_depth   # (480, 640) float32, meters
obs["wrist_rgb"]   = self._wrist_rgb    # (480, 640, 3) uint8
obs["wrist_depth"] = self._wrist_depth  # (480, 640) float32
```

> 渲染器在主线程（物理循环），结果写入 numpy 数组后由 ZMQ 线程读取，遵循与 `_joint_state` 相同的无锁共享模式。

### 与 gello_diffusion.py 的对接

`gello_diffusion.py` 当前检查 `BASE_IMAGE_KEY = "base_rgb"`，新环境渲染后直接填充此字段，**无需修改转换脚本**。

---

## gello_diffusion.py 说明

**它是什么**：`gello/data_utils/gello_diffusion.py` 是 **pickle → zarr 格式转换工具**，不包含 diffusion policy 模型或训练代码。

**功能**：
1. 扫描 `./bc_data/pick_cube/` 下的 episode 目录，加载 `.pkl` 文件
2. 过滤掉帧数 ≤ 30 的 episode
3. 将每帧数据转为结构化 zarr 数组：

| zarr 数组 | Shape | 说明 |
|-----------|-------|------|
| `data/img` | (N, 3, 180, 320) uint8 | `base_rgb` 缩放后 |
| `data/state` | (N, 14) float32 | `joint_positions(6)` + `ee_pos_quat(7)` + `gripper(1)` |
| `data/action` | (N, 7) float32 | `control`（6 臂 + 1 夹爪） |
| `meta/episode_ends` | (E,) int64 | episode 边界索引 |

4. 使用 zstd 压缩（Blosc），存为 `./bc_data/pick_cube.zarr`
5. 输出可直接被标准 `ReplayBuffer`（如 diffusion policy 框架中的实现）加载

**不包含**：diffusion policy 模型定义、噪声调度、训练循环、推理代码。

---

## 项目中 diffusion policy 相关代码现状

| 组件 | 是否存在 |
|------|---------|
| 数据采集（run_env.py + gello） | ✅ |
| pickle → zarr 转换（gello_diffusion.py） | ✅ |
| pickle → HDF5 转换（demo_to_gdict.py） | ✅ |
| 简单 BC 训练（simple_bc/train.py） | ✅（独立可用） |
| **Diffusion policy 模型/训练代码** | ❌ 不存在 |
| **Diffusion policy 推理代码** | ❌ 不存在 |

**结论**：项目提供了 diffusion policy 训练所需的**数据管道**（采集 → 转换 → zarr），但**训练本身**需对接外部库（如 [diffusion_policy](https://github.com/real-robot-lab/diffusion_policy) 或 [Lerobot](https://github.com/huggingface/lerobot)）。该 zarr 格式与主流实现兼容。

---

## 完整 observations 字段（更新后）

| 字段 | Shape | 必填 by gello_diffusion |
|------|-------|------------------------|
| `joint_positions` | (7,) | ✅（取前6位） |
| `joint_velocities` | (7,) | — |
| `ee_pos_quat` | (7,) | ✅ |
| `gripper_position` | scalar | ✅ |
| `cube_pos` | (3,) | — （可扩展 state） |
| `cube_quat` | (4,) | — （可扩展 state） |
| **`base_rgb`** | **(480,640,3)** | **✅ 必须** |
| **`base_depth`** | **(480,640)** | — |
| **`wrist_rgb`** | **(480,640,3)** | — |
| **`wrist_depth`** | **(480,640)** | — |

---

## Verification

1. `python experiments/launch_nodes.py --robot sim_ur_cube`  
   → Mujoco 窗口出现，场景含地板 + UR5+夹爪 + 红色立方体 + 两个相机视锥
2. `python experiments/run_env.py --agent=gello`  
   → "This robot has 7 dofs"，gello 实时控制机械臂
3. 采集 1 个 episode 后，加载 `.pkl` 验证：
   - 含 `base_rgb`（480×640 RGB 图像）
   - 含 `wrist_rgb`（腕部视角）
   - 含 `cube_pos`（立方体位置）
4. 运行 `python gello/data_utils/gello_diffusion.py` → 生成 `bc_data/pick_cube.zarr`
