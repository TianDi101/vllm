# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv half of the ROCm Kimi-K3 prefill checkpoint."""

from dataclasses import dataclass, replace

import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.kda_checkpoint import (
    FlashKDAPrefillCheckpointExporter,
)


@dataclass(frozen=True)
class KimiK3ROCmKDAPrefillCheckpointExporter(FlashKDAPrefillCheckpointExporter):
    """Conv window only; `fused_kda_chunk` writes the recurrent half itself.

    `state_len` is `conv_kernel_size - 1`, not the spec-widened conv row.
    """

    def export(  # type: ignore[override]
        self,
        checkpoint: MambaPrefillCheckpointMetadata,
        *,
        raw_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> None:
        assert self.state_len is not None
        offsets = checkpoint.checkpoint_offsets
        if offsets.dim() == 2:
            # `[N, K]` (tail window): one program per entry, each reading its
            # own sequence's query start.
            width = offsets.shape[1]
            checkpoint = replace(
                checkpoint,
                checkpoint_offsets=offsets.reshape(-1),
                state_indices=checkpoint.state_indices.reshape(-1),
            )
            cu_seqlens = cu_seqlens[: offsets.shape[0]].repeat_interleave(width)
        # An empty recurrent checkpoint masks the kernel's recurrent half off.
        super().export(
            checkpoint,
            raw_qkv=raw_qkv,
            conv_state=conv_state,
            recurrent_checkpoint=conv_state[:, :0],
            recurrent_state=conv_state,
            cu_seqlens=cu_seqlens,
        )
