# Ascend C Tiling And Grid Pack

## Block 级

- AIV 适合纯 Vector，AIC 适合 Cube/Matmul，MIX kernel 需要考虑物理 core group。
- 核数必须从平台接口或 active environment 获取，例如 AIV/AIC core count；不要复制设备常量。
- 小 shape 不强行铺满所有核，避免 scalar 调度和 tail 开销超过收益。
- `formerNum * formerLength + tailNum * tailLength` 或等价字段必须覆盖全部数据且无重叠。
- GM offset、shape size、字节乘积使用 64 位整型。

## UB 级

- UB tile 计算要包含输入、输出、临时量、FP32 buffer、index/mask、double buffer。
- `tileLength` 是容量上限，`curTileLength` 是本轮有效长度，两者不可混用。
- Vector API count、CopyOut 有效长度用 `curTileLength`。
- UB offset 和 buffer size 用 32B 对齐长度。

## L2 / 大 Shape

- input + output 明显超过 L2 时，优先检查 L2 split 或按复用方向切分。
- 分核要兼顾连续 GM 访问和负载均衡；不要为了均衡制造大量非连续小 DMA。

## Tail 检查

- block tail 和 tile tail 分开处理。
- padding 值不能参与有效结果。
- 对齐长度只用于搬运和 UB 布局，不用于语义计算。
