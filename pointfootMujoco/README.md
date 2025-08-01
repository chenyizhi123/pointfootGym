# Mujoco Sim-to-Sim

## Install the low level controller

Linux x86_64 environment

```bash
pip install pointfoot-sdk-lowlevel/python3/amd64/limxsdk-*-py3-none-any.whl
```

## Mujoco simulator

```bash
export ROBOT_TYPE=PF_TRON1A  # choose robot model
python simulator.py
```

## RL controller

```bash
export ROBOT_TYPE=PF_TRON1A  # choose robot model
python rl_controller.py
```
