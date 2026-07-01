换成 **Dense AWQ 推理期 W4A16 Linear**。

推荐目标就是替代这里的调用：

```python
torch_npu.npu_weight_quant_batchmatmul(
    x=x,
    weight=layer.weight,
    antiquant_scale=layer.weight_scale,
    antiquant_offset=layer.weight_offset,
    antiquant_group_size=layer.awq_runtime_group_size,
    bias=bias,
)
```

**先固定一个最小场景**

只做这个场景：

```text
Dense AWQ W4A16, zero_point=True, group_size=128, bias=None, fp16 activation
```

**输入约束**

```text
x:
  dtype: torch.float16
  shape: [M, K]
  device: npu
  layout: contiguous

weight:
  dtype: torch.int32
  shape: [K, N / 8]
  device: npu
  layout: Ascend int4pack, contiguous

antiquant_scale:
  dtype: torch.float16
  shape: [K / 128, N]
  device: npu
  layout: contiguous

antiquant_offset:
  dtype: torch.float16
  shape: [K / 128, N]
  device: npu
  layout: contiguous

antiquant_group_size:
  value: 128

bias:
  value: None
```

**输出约束**

```text
output:
  dtype: torch.float16
  shape: [M, N]
  device: npu
```

**语义**

```text
output[m, n] =
  sum_k x[m, k] * (int4_weight[k, n] + antiquant_offset[k // 128, n])
                  * antiquant_scale[k // 128, n]
```

其中 `int4_weight` 是 `weight` 解出的 signed int4 值。

**建议首个 benchmark shape**

```text
M = 16
K = 4096
N = 22016
group_size = 128
```

这个更接近 Dense MLP gate/up projection，算量足够大，能看出 agent 是否真的优化到了核心计算。

**正确性参考**

直接用当前 vendor op 做 reference：

```python
ref = torch_npu.npu_weight_quant_batchmatmul(
    x=x,
    weight=weight,
    antiquant_scale=scale,
    antiquant_offset=offset,
    antiquant_group_size=128,
    bias=None,
)

out = candidate_awq_w4a16_linear(x, weight, scale, offset)

torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
```

**性能指标**

只看这个：

```text
candidate_awq_w4a16_linear / npu_weight_quant_batchmatmul
```

用 `torch.npu.Event` 计时，warmup 100 次，正式 1000 次。
