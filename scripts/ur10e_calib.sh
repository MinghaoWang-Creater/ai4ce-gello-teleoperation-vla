#!/bin/bash

# Calibration

python scripts/gello_get_offset.py \
    --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U0XG-if00-port0

# then go to  gello/agents/gello_agent.py 100 line
#    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U0XG-if00-port0": DynamixelRobotConfig(
#         joint_ids=(1, 2, 3, 4, 5, 6),
#         joint_offsets=[2*np.pi/2, 4*np.pi/2, 3*np.pi/2, 5*np.pi/2, 4*np.pi/2, 4*np.pi/2], # modify this line
#         joint_signs=(1, 1, -1, 1, 1, 1),
#         gripper_config=(7, 20, -22), # modify this line
#     )
# }