import pytest
import torch

from ventris.common import mixed_precision


def test_mixed_precision_uses_native_cuda_bfloat16(monkeypatch):
    autocast_calls = []
    support_checks = []

    def supports_bfloat16(*, including_emulation):
        support_checks.append(including_emulation)
        return True

    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        supports_bfloat16,
    )
    monkeypatch.setattr(
        torch,
        "autocast",
        lambda device_type, *, dtype: autocast_calls.append((device_type, dtype)),
    )

    mixed_precision(torch.device("cpu"))
    mixed_precision(torch.device("cuda"))

    assert support_checks == [False]
    assert autocast_calls == [("cuda", torch.bfloat16)]


def test_mixed_precision_uses_fp32_without_native_bfloat16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: False)
    monkeypatch.setattr(
        torch,
        "autocast",
        lambda *args, **kwargs: pytest.fail("autocast should be disabled"),
    )

    with mixed_precision(torch.device("cuda")):
        pass
