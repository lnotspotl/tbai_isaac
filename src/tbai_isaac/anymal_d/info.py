#!/usr/bin/env python3

import os

URDF_PATH = os.path.join(os.path.dirname(__file__), "resources/urdf/anymal.urdf")
URDF_FOLDER = os.path.join(os.path.dirname(__file__), "resources")

BASE_NAME = "base"
FOOT_NAMES = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]
