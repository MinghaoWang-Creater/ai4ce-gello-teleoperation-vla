#!/bin/bash

# Calibration
# source .venv/bin/activate
# python scripts/gello_get_offset.py \
#     --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
#     --joint-signs 1 1 -1 1 1 1 \
#     --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U0XG-if00-port0

SESSION_NAME="ur10e_data_collection"

# # Camera Node
# tmux new-session -d -s $SESSION_NAME -n camera
# tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
# tmux send-keys -t $SESSION_NAME "PID=\$(lsof -ti:5001); [ -n \"\$PID\" ] && kill \$PID" C-m
# tmux send-keys -t $SESSION_NAME "python experiments/launch_camera_nodes.py" C-m

# UR10e Node
tmux new-window -t $SESSION_NAME -n ur10e
tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "PID=\$(lsof -ti:6001); [ -n \"\$PID\" ] && kill \$PID" C-m
tmux send-keys -t $SESSION_NAME "python experiments/launch_nodes.py --robot=ur --robot_ip=192.168.50.144 --robot_port=6001" C-m

# Gello Agent
tmux new-window -t $SESSION_NAME -n gello
tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python experiments/run_env.py --agent=gello --robot_port=6001 --data_dir=./ur10e_data  --use_save_interface" C-m

tmux attach -t $SESSION_NAME