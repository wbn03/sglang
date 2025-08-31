import pytest
import torch
from sgl_kernel import meanpooling


def ref_meanpooling(
    k_cache, comressed_k_cache, begin_pos, compressed_len, kernel_size, stride
):
    batch_num = k_cache.shape[0]
    max_seqlen = k_cache.shape[1]
    k_cache = k_cache.view(batch_num, max_seqlen, -1)
    for batch_id in range(batch_num):
        cur_len = compressed_len[batch_id]
        cur_pos = begin_pos[batch_id]
        cur_ori_cache_len = (cur_len - 1) * stride + kernel_size
        cur_k_cache = k_cache[
            batch_id, cur_pos * stride : cur_pos * stride + cur_ori_cache_len, :
        ].permute(1, 0)
        cur_k_cache = cur_k_cache.unsqueeze(0)
        cur_res = torch.nn.functional.avg_pool1d(
            cur_k_cache, kernel_size, stride, count_include_pad=False
        )
        cur_res = cur_res.squeeze(0)
        cur_res = cur_res.permute(1, 0)
        comressed_k_cache[batch_id, cur_pos : cur_pos + cur_len, :] = cur_res

    return comressed_k_cache


@pytest.mark.parametrize("batch_num", [32])
@pytest.mark.parametrize("max_seqlen", [2048])
@pytest.mark.parametrize("dim", [128 * 8])
@pytest.mark.parametrize("kernel_size", [32])
@pytest.mark.parametrize("stride", [16])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_meanpooling(batch_num, max_seqlen, dim, kernel_size, stride, dtype):
    k_cache = torch.randn(batch_num, max_seqlen, dim, device="cuda").to(dtype)
    compress_cache_len = (max_seqlen + stride - 1) // stride
    comressed_k_cache = torch.randn(
        batch_num, compress_cache_len, dim, device="cuda"
    ).to(dtype)
    comressed_k_cache_ref = comressed_k_cache
    begin_pos = torch.randint(
        0, (max_seqlen - kernel_size) // stride // 2, (batch_num,)
    ).to(torch.int32)
    compressed_len = torch.randint(1, max_seqlen // 2 // stride, (batch_num,)).to(
        torch.int32
    )
    comressed_k_cache_ref = ref_meanpooling(
        k_cache, comressed_k_cache_ref, begin_pos, compressed_len, kernel_size, stride
    )
    comressed_k_cache = meanpooling(
        k_cache,
        comressed_k_cache,
        begin_pos,
        compressed_len,
        begin_pos.cuda(),
        compressed_len.cuda(),
        kernel_size=kernel_size,
        stride=stride,
    )
    assert torch.allclose(comressed_k_cache_ref, comressed_k_cache, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
