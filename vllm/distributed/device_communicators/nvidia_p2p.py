# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

NVIDIA_DRIVER_PARAMS_PATH = "/proc/driver/nvidia/params"
P2P_OVERRIDE_REMEDY = (
    'options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;'
    'RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"'
)


def p2p_override_state(path: str = NVIDIA_DRIVER_PARAMS_PATH) -> bool | None:
    """Whether the NVIDIA driver runs with the ForceP2P registry override.

    Returns None when the driver parameters cannot be read, so callers never
    act on a guess.
    """
    try:
        with open(path) as params:
            lines = params.read().splitlines()
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip() == "RegistryDwords":
            return "ForceP2P=" in value
    return False


def require_p2p_override() -> bool:
    return os.getenv("VLLM_PCIE_REQUIRE_P2P_OVERRIDE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
