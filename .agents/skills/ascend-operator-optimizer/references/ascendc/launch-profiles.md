# Ascend C Launch Profiles

AscendC 不应只按一个大类处理。优化前先声明 `implementation.launch_profile`，并决定是否需要 launch probe。

## Profile 类型

| launch_profile | 典型路径 | 必查项 |
| --- | --- | --- |
| `ascendc-standard` | `ascend-kernel` / `EXEC_KERNEL_CMD` | `GetBlockIdx()` 预期为 `0..blockDim-1`，block tiling 覆盖无遗漏无重叠 |
| `ascendc-msopgen-aclnn-dynamic` | msopgen + aclnn custom op | 必须先跑 launch coverage probe；不要默认普通多 block 语义 |
| `ascendc-cube-matmul` | Cube/Matmul/Fixpipe 路径 | AIC、L0A/L0B/L0C、L1、BT/FP buffer、AtomicAdd |
| `ascendc-vector` | AIV/Vector/UB 路径 | UB、DataCopyPad、TQue、Counter、double buffer |

## Launch Coverage Probe

`launch_probe` 命令必须走目标项目真实 launch 路径，而不是单独调用一个不同的测试入口。

推荐 probe：

```text
1. 编译一个 constant-fill kernel。
2. 设置 blockDim > 1。
3. 每个 block 写不同区间或全写 1。
4. 检查输出覆盖率是否 100%。
5. 记录 GetBlockIdx/GetBlockNum 可观察行为。
```

命令通过环境变量写 JSON：

```text
$OPT_HARNESS_PROBE_RESULT
```

最小 JSON：

```json
{
  "name": "launch_coverage_probe",
  "status": "pass",
  "launch_profile": "ascendc-msopgen-aclnn-dynamic",
  "block_dim": 8,
  "coverage": 1.0,
  "observed_block_idx": [0, 1, 2, 3, 4, 5, 6, 7],
  "observed_block_num": 8,
  "notes": []
}
```

如果 probe 失败，应停止性能优化。常见结论是当前 launch path 不支持普通多 block 逻辑编号，需先换 launch path、工程路径或 block 分配策略。
