import numpy as np
from gello.robots.dynamixel import DynamixelRobot # 假设你使用的是标准 GELLO 库

# --- 请根据你的实际情况修改以下参数 ---
PORT = '/dev/ttyUSB0'  # Windows 可能是 'COM3'
BAUDRATE = 57600       # 常见为 57600 或 1000000
JOINT_IDS = [1, 2, 3, 4, 5, 6, 7] # 你在 Wizard 中看到的 6 个电机 ID
# ----------------------------------

try:
    # 初始化机器人硬件接口
    robot = DynamixelRobot(port=PORT, baudrate=BAUDRATE, joint_ids=JOINT_IDS)
    print(f"成功连接到 GELLO！端口: {PORT}")

    print("现在请手动移动 GELLO 手柄，程序将实时打印关节角度（弧度）...")
    print("按 Ctrl+C 退出")

    while True:
        # 读取当前关节位置
        joints = robot.get_joint_state()
        # 格式化输出，保留三位小数
        formatted_joints = [f"{j:.3f}" for j in joints]
        print(f"当前角度: {formatted_joints}", end='\r')

except Exception as e:
    print(f"\n连接或读取失败: {e}")
    print("提示：请检查端口名是否正确，以及 Dynamixel Wizard 是否已关闭。")