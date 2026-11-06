# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import pathlib

import pytest

_MODULE = (
    pathlib.Path(__file__).parents[2]
    / "vllm/distributed/device_communicators/nvidia_p2p.py"
)
_spec = importlib.util.spec_from_file_location("nvidia_p2p", _MODULE)
nvidia_p2p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nvidia_p2p)


def _params(tmp_path, registry_line):
    path = tmp_path / "params"
    path.write_text("ResmanDebugLevel: 4294967295\n" + registry_line + "\nPreserveVideoMemoryAllocations: 0\n")
    return str(path)


def test_override_active(tmp_path):
    line = 'RegistryDwords: "ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2"'
    assert nvidia_p2p.p2p_override_state(_params(tmp_path, line)) is True


def test_override_empty_registry(tmp_path):
    assert nvidia_p2p.p2p_override_state(_params(tmp_path, 'RegistryDwords: ""')) is False


def test_override_registry_with_other_keys(tmp_path):
    line = 'RegistryDwords: "EnableResizableBar=1"'
    assert nvidia_p2p.p2p_override_state(_params(tmp_path, line)) is False


def test_override_line_absent(tmp_path):
    path = tmp_path / "params"
    path.write_text("ResmanDebugLevel: 0\n")
    assert nvidia_p2p.p2p_override_state(str(path)) is False


def test_unreadable_returns_none(tmp_path):
    assert nvidia_p2p.p2p_override_state(str(tmp_path / "missing")) is None


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), (" On ", True), ("0", False), ("", False), ("no", False)],
)
def test_require_flag(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_PCIE_REQUIRE_P2P_OVERRIDE", value)
    assert nvidia_p2p.require_p2p_override() is expected


def test_require_flag_unset(monkeypatch):
    monkeypatch.delenv("VLLM_PCIE_REQUIRE_P2P_OVERRIDE", raising=False)
    assert nvidia_p2p.require_p2p_override() is False
