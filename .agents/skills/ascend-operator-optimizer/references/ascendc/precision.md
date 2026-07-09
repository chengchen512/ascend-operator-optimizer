# Ascend C Precision Pack

正确性失败时不进入 benchmark。性能优化前先让精度问题归零或回退本轮修改。

## 指标

若目标仓库没有更严格标准，AscendC 精度评估可记录：

- MERE：最大相对误差类指标。
- MARE：平均相对误差类指标。

参考阈值：

| dtype | MERE 阈值 | MARE 阈值 |
| --- | ---: | ---: |
| fp16 | `2^-10` | `10 * 2^-10` |
| bf16 | `2^-7` | `10 * 2^-7` |
| fp32 | `2^-13` | `10 * 2^-13` |
| HiFloat32 | `2^-11` | `10 * 2^-11` |
| float8 e4m3 | `2^-3` | `10 * 2^-3` |
| float8 e5m2 | `2^-2` | `10 * 2^-2` |

## 症状定位

| 现象 | 优先排查 |
| --- | --- |
| FP32 通过，FP16/BF16 失败 | 缺少 FP32 中间精度或 Cast round mode |
| 全 0 或全常数 | CopyOut 偏移、输出 GM、AtomicNone |
| NaN/Inf | 除零、非法 domain、exp overflow、padding 参与计算 |
| 首错在 tile 边界 | CopyIn/CopyOut 偏移、`curTileLength` |
| 只在 tail 失败 | tail 长度、padding、CopyOut 有效长度 |
| 多次运行不同 | MTE 异步同步、队列配对、未初始化 |
| 单核通过，多核失败 | block tiling 重叠/遗漏、workspace 分区、atomic 初始化 |

## MSSanitizer

适合用作 probe：

- `memcheck`：非法访问、越界、泄漏。
- `racecheck`：数据竞争。
- `initcheck`：未初始化。
- `synccheck`：同步错误。

`--check-device-heap` 和 `--check-cann-heap` 分开跑，不要同时开启。空日志通常只表示该工具未发现问题，不代表精度必然正确。
