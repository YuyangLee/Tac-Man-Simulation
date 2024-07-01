import numpy as np

# Franka Panda

CONTACT_AREAS = {
    "panda": {
        "L": [[-0.008787, 0.000071, 0.036023 ], [ 0.008787, 0.000071, 0.036023 ], [ 0.008787, 0.000071, 0.053879], [ -0.008787, 0.000071, 0.053879] ],
        "R": [[-0.008787, 0.000071, 0.036023 ], [ 0.008787, 0.000071, 0.036023 ], [ 0.008787, 0.000071, 0.053879], [ -0.008787, 0.000071, 0.053879] ]
    }
}

ARM_DOF_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
ARM_DOF_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
ARM_DOF_RG = ARM_DOF_HI - ARM_DOF_LO

ARM_JOINTS = [ "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7" ]
GRIPPER_JOINTS = [ "panda_finger_joint1", "panda_finger_joint2" ]

HAND_NAME = "panda"

# Simulation control
MAX_STEPS = 50000

STATE_PROC, STATE_RECV, STATE_SUCC = 0, 1, 2

CONTACT_THRES = 0.005

# Algorithm params

delta_0 = 0.0004
alpha = 0.6
