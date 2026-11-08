"""Fused CUDA carry for the FP32 modal phase accumulator."""

import torch
import triton
import triton.language as tl


@triton.jit
def _carry(Totals, Starts, count):
    row = tl.program_id(0) * count
    state = 0.0
    compensation = 0.0
    for index in range(count):
        tl.store(Starts + row + index, state)
        total = tl.load(Totals + row + index)
        increment = (total - tl.floor(total)) - compensation
        updated = state + increment
        compensation = (updated - state) - increment
        state = updated - tl.floor(updated)


def wrapped_block_starts(totals):
    totals = totals.contiguous()
    starts = torch.empty_like(totals)
    count = totals.shape[-1]
    _carry[(totals.numel() // count,)](
        totals, starts, count, num_warps=1, enable_fp_fusion=False
    )
    return starts
