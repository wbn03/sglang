from typing import Optional

import torch


def meanpooling(
    k_cache: torch.Tensor,
    compressed_cache: torch.Tensor,
    begin_pos: torch.Tensor,
    compressed_len: torch.Tensor,
    cu_begin_pos: Optional[torch.Tensor] = None,
    cu_compressed_len: Optional[torch.Tensor] = None,
    kernel_size: int = 32,
    stride: int = 16,
) -> torch.Tensor:
    r"""Meanpooling.

    Parameters
    ----------
    k_cache: torch.Tensor
        Input tensor, shape (batch_size, max_seqlen, hidden_size).
    compressed_cache: torch.Tensor
        Weight tensor, shape (batch_size, max_compressed_len, hidden_size).
    begin_pos: torch.Tensor
        The position of each sequence in cache.
    compressed_len: torch.Tensor
        The compressed cache len of each sequence.
    cu_begin_pos: Optional[torch.Tensor]
        The position of each sequence in cache on cuda.
    cu_compressed_len: Optional[torch.Tensor]
        The compressed cache len of each sequence on cuda.
    kernel_size: int
        Pooling kernel size.
    stride: int
        Pooling stride.

    Returns
    -------
    output: torch.Tensor
        Cache after compressed.
    """
    out = torch.ops.sgl_kernel.meanpooling.default(
        k_cache,
        compressed_cache,
        begin_pos,
        compressed_len,
        cu_begin_pos if cu_begin_pos is not None else None,
        cu_compressed_len if cu_begin_pos is not None else None,
        kernel_size=kernel_size,
        stride=stride,
    )
    return out
