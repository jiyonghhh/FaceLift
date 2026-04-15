"""Blackwell (sm_120) compatibility shim for xformers.

The bundled xformers flash-attention Hopper kernel fails with
"invalid argument" on RTX 50-series (compute capability 12.0).
This module monkey-patches xformers so any caller asking for the
flash op (or the default dispatch) transparently uses the cutlass
backend, which works on Blackwell.

Import this module BEFORE importing anything that uses xformers
(diffusers UNet, mvdiffusion transformers, gslrm transformers).
"""
from __future__ import annotations

import torch


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 12


def patch_xformers_for_blackwell() -> None:
    if not _is_blackwell():
        return

    import xformers.ops as xops
    import xformers.ops.fmha as fmha

    cutlass_fw = fmha.cutlass.FwOp
    cutlass_bw = fmha.cutlass.BwOp

    fmha.flash.FwOp = cutlass_fw
    fmha.flash.BwOp = cutlass_bw

    original_mea = xops.memory_efficient_attention

    def _mea(query, key, value, attn_bias=None, p=0.0, scale=None, op=None, **kw):
        if op is None:
            op = (cutlass_fw, cutlass_bw)
        else:
            fw, bw = op
            if "flash" in fw.__module__:
                fw = cutlass_fw
            if "flash" in bw.__module__:
                bw = cutlass_bw
            op = (fw, bw)
        return original_mea(query, key, value, attn_bias=attn_bias, p=p, scale=scale, op=op, **kw)

    xops.memory_efficient_attention = _mea
    fmha.memory_efficient_attention = _mea


patch_xformers_for_blackwell()
