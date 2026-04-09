import pickle
import threading
import time
from typing import Any, Dict, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from dm_control import mjcf

from gello.robots.sim_robot import ZMQRobotServer, ZMQServerThread, attach_hand_to_arm

# 放置区中心（固定）
GOAL_POS = np.array([0.35, 0.25, 0.0])
# 放置区方框半边长（米）
GOAL_HALF = 0.07

# 立方体随机初始化范围：x ∈ [center_x ± rx], y ∈ [center_y ± ry]
CUBE_CENTER = np.array([0.35, 0.0])
CUBE_RANGE  = np.array([0.08, 0.08])


def _add_goal_zone(worldbody, cx: float, cy: float, half: float = GOAL_HALF):
    """在地面添加绿色方框标记放置区（4 根薄条）。"""
    color = [0.05, 0.85, 0.15, 1.0]
    bar_w = 0.008   # 条宽（半）
    bar_h = 0.002   # 条高（半）
    z = bar_h       # 贴地

    # 上下两条（沿 X 方向）
    for name, y_off in [("goal_n", cy + half), ("goal_s", cy - half)]:
        worldbody.add("geom", name=name, type="box",
                      pos=[cx, y_off, z],
                      size=[half + bar_w, bar_w, bar_h],
                      rgba=color, contype=0, conaffinity=0)
    # 左右两条（沿 Y 方向）
    for name, x_off in [("goal_e", cx + half), ("goal_w", cx - half)]:
        worldbody.add("geom", name=name, type="box",
                      pos=[x_off, cy, z],
                      size=[bar_w, half - bar_w, bar_h],
                      rgba=color, contype=0, conaffinity=0)


def build_scene_with_cube(
    robot_xml_path,
    gripper_xml_path=None,
    cube_size: float = 0.025,
    cube_init_pos: Tuple[float, float, float] = (0.35, 0.0, 0.025),
    cube_rgba: Tuple = (1.0, 0.2, 0.1, 1.0),
):
    arena = mjcf.RootElement()

    # 单盏聚光灯，照射工作台（方块区 + 放置区之间）
    arena.worldbody.add(
        "light",
        name="work_light",
        pos=[0.35, 0.12, 1.4],
        dir=[0.0, -0.1, -1.0],
        diffuse=[0.55, 0.55, 0.55],
        specular=[0.1, 0.1, 0.1],
        castshadow=True,
        cutoff=45,
    )

    # 地板
    arena.worldbody.add(
        "geom",
        name="floor",
        type="plane",
        size=[2, 2, 0.1],
        pos=[0, 0, 0],
        rgba=[0.75, 0.75, 0.75, 1.0],
        friction=[0.8, 0.005, 0.0001],
    )

    # 放置区绿色方框（contype=0 不参与碰撞）
    _add_goal_zone(arena.worldbody, GOAL_POS[0], GOAL_POS[1])

    # 机械臂
    arm_mjcf = mjcf.from_path(str(robot_xml_path))

    # 腕部相机（attach 后名称变为 "ur5e/wrist_cam"）
    wrist_body = arm_mjcf.find("body", "wrist_3_link")
    if wrist_body is not None:
        wrist_body.add(
            "camera",
            name="wrist_cam",
            pos=[0.0, 0.0, -0.08],
            euler=[0, 180, 0],
        )

    if gripper_xml_path is not None:
        gripper_mjcf = mjcf.from_path(str(gripper_xml_path))
        attach_hand_to_arm(arm_mjcf, gripper_mjcf)

    arena.worldbody.attach(arm_mjcf)

    # 全局基础相机
    arena.worldbody.add(
        "camera",
        name="base_cam",
        pos=[1.2, 0.0, 1.0],
        xyaxes=[0, 1, 0, -0.6, 0, 0.8],
    )

    # 立方体（自由体，6 DOF）
    cube_body = arena.worldbody.add("body", name="cube", pos=list(cube_init_pos))
    cube_body.add("freejoint", name="cube_joint")
    cube_body.add(
        "geom",
        type="box",
        size=[cube_size] * 3,
        rgba=list(cube_rgba),
        contype=1,
        conaffinity=1,
        mass=0.1,
    )

    return arena


# ---------------------------------------------------------------------------
# 扩展 ZMQ 服务器，支持 reset_episode 指令
# ---------------------------------------------------------------------------

class GraspZMQRobotServer(ZMQRobotServer):
    """在标准 ZMQ 协议基础上追加 reset_episode 方法。"""

    _robot: "MujocoGraspServer"  # narrow type for type checkers

    def serve(self) -> None:
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)
                method = request.get("method")
                args = request.get("args", {})

                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "reset_episode":
                    result = self._robot.reset_episode()
                else:
                    result = {"error": f"Invalid method: {method}"}

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                pass


# ---------------------------------------------------------------------------
# 主仿真服务器
# ---------------------------------------------------------------------------

class MujocoGraspServer:
    def __init__(
        self,
        xml_path,
        gripper_xml_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        cube_size: float = 0.025,
        cube_center: np.ndarray = CUBE_CENTER,
        cube_range: np.ndarray = CUBE_RANGE,
    ):
        self._has_gripper = gripper_xml_path is not None
        self._cube_size = cube_size
        self._cube_center = np.array(cube_center)
        self._cube_range = np.array(cube_range)

        # 初始随机位置
        init_xy = self._sample_cube_xy()
        init_pos = (init_xy[0], init_xy[1], cube_size)

        arena = build_scene_with_cube(xml_path, gripper_xml_path,
                                      cube_size=cube_size, cube_init_pos=init_pos)

        assets: Dict[str, Any] = {}
        for asset in arena.asset.all_children():
            if asset.tag == "mesh":
                f = asset.file
                assets[f.get_vfs_filename()] = asset.file.contents

        xml_string = arena.to_xml_string()
        with open("arena_grasp.xml", "w") as f:
            f.write(xml_string)

        self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        self._data = mujoco.MjData(self._model)

        self._num_joints = self._model.nu
        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = self._joint_state.copy()

        # 找到 cube freejoint 在 qpos/qvel 中的偏移地址
        cube_jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self._cube_qpos_adr = int(self._model.jnt_qposadr[cube_jid])
        self._cube_qvel_adr = int(self._model.jnt_dofadr[cube_jid])

        # 待执行重置标志（由 reset_episode() 在 ZMQ 线程设置，physics 循环消费）
        self._reset_flag = threading.Event()

        # offscreen 渲染器
        self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        self._depth_renderer = mujoco.Renderer(self._model, height=480, width=640)
        self._depth_renderer.enable_depth_rendering()

        self._base_rgb   = np.zeros((480, 640, 3), dtype=np.uint8)
        self._base_depth = np.zeros((480, 640), dtype=np.float32)
        self._wrist_rgb  = np.zeros((480, 640, 3), dtype=np.uint8)
        self._wrist_depth= np.zeros((480, 640), dtype=np.float32)

        self._zmq_server = GraspZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

        self._print_joints = print_joints

    # ------------------------------------------------------------------

    def _sample_cube_xy(self) -> np.ndarray:
        lo = self._cube_center - self._cube_range
        hi = self._cube_center + self._cube_range
        return np.random.uniform(lo, hi)

    def _apply_cube_reset(self) -> None:
        """在 physics 循环内执行立方体位置随机重置（线程安全）。"""
        xy = self._sample_cube_xy()
        z  = self._cube_size
        a  = self._cube_qpos_adr
        # 位置
        self._data.qpos[a:a+3] = [xy[0], xy[1], z]
        # 姿态（单位四元数：w,x,y,z）
        self._data.qpos[a+3:a+7] = [1.0, 0.0, 0.0, 0.0]
        # 清零速度
        va = self._cube_qvel_adr
        self._data.qvel[va:va+6] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def reset_episode(self) -> str:
        """由 ZMQ 客户端调用，请求在下一物理步重置立方体。"""
        self._reset_flag.set()
        return "ok"

    # ------------------------------------------------------------------

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints
        if self._has_gripper:
            _js = joint_state.copy()
            _js[-1] = _js[-1] * 255
            self._joint_cmd = _js
        else:
            self._joint_cmd = joint_state.copy()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions  = self._data.qpos.copy()[: self._num_joints]
        joint_velocities = self._data.qvel.copy()[: self._num_joints]

        ee_site = "attachment_site"
        try:
            ee_pos = self._data.site_xpos.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_mat = self._data.site_xmat.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat)
        except Exception:
            ee_pos  = np.zeros(3)
            ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

        gripper_pos = self._data.qpos.copy()[self._num_joints - 1]

        cube_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_pos = self._data.xpos[cube_id].copy()
        cube_quat= self._data.xquat[cube_id].copy()

        return {
            "joint_positions":  joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat":      np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
            "cube_pos":         cube_pos,
            "cube_quat":        cube_quat,
            "goal_pos":         GOAL_POS.copy(),
            "base_rgb":         self._base_rgb,
            "base_depth":       self._base_depth,
            "wrist_rgb":        self._wrist_rgb,
            "wrist_depth":      self._wrist_depth,
        }

    def _render_cameras(self) -> None:
        self._renderer.update_scene(self._data, camera="base_cam")
        self._base_rgb = self._renderer.render().copy()

        self._depth_renderer.update_scene(self._data, camera="base_cam")
        self._base_depth = self._depth_renderer.render().copy()

        try:
            self._renderer.update_scene(self._data, camera="ur5e/wrist_cam")
            self._wrist_rgb = self._renderer.render().copy()

            self._depth_renderer.update_scene(self._data, camera="ur5e/wrist_cam")
            self._wrist_depth = self._depth_renderer.render().copy()
        except Exception:
            pass

    def serve(self) -> None:
        self._zmq_server_thread.start()
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # episode 重置（ZMQ 线程通过 reset_episode() 触发）
                if self._reset_flag.is_set():
                    self._apply_cube_reset()
                    self._reset_flag.clear()

                self._data.ctrl[:] = self._joint_cmd
                mujoco.mj_step(self._model, self._data)
                self._joint_state = self._data.qpos.copy()[: self._num_joints]

                self._render_cameras()

                if self._print_joints:
                    print(self._joint_state)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        self._data.time % 2
                    )

                viewer.sync()

                time_until_next_step = self._model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()
