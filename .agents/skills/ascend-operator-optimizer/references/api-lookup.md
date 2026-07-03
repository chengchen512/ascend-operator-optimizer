# Ascend C API Lookup Guide

本文件定义 agent 查找 Ascend C API 的高效流程。目标是避免把 `docs/api` 的 2000 多个 Markdown 文件一次性读入上下文，同时在改代码前能查到精确接口、参数单位、重载限制和示例。

## 总原则

1. 先读本 Skill 的蒸馏文档，明确问题类型：
   - 搬运：`ascendc/data-copy.md`
   - 队列/资源/API 结构：`ascendc/api-usage.md`
   - UB/L1/L0/workspace：`ascendc/memory.md`
   - 流水：`ascendc/pipeline.md`
   - 精度/同步错误：`ascendc/precision.md`
2. 需要确认 API 签名、参数单位、平台约束或示例时，再查官方 API 快照。
3. 不全量读取 `reference/asc-devkit-api-9.0.0/docs/api`。先用文件名和关键字检索，只打开最相关的 1 到 3 个文档。
4. 如果官方 API 快照找不到某个 overload，不要凭记忆编造。记录“未找到可用文档证据”，再用目标 CANN 环境编译或头文件验证。

## API 快照位置

在本仓库中，官方 API 快照位于：

```text
reference/asc-devkit-api-9.0.0/docs/api
```

若 Skill 被复制到别的目标仓库，快照可能不存在。查找顺序：

1. 环境变量 `ASCENDC_API_DOCS_ROOT` 指向的目录。
2. 当前仓库根目录下的 `reference/asc-devkit-api-9.0.0/docs/api`。
3. 当前 Skill 所在仓库根目录下的同名路径。
4. 目标 CANN 安装或官方在线文档。

## 推荐工具

优先使用随 Skill 提供的轻量检索脚本：

```bash
python .agents/skills/ascend-operator-optimizer/scripts/api_lookup.py DataCopyPad
python .agents/skills/ascend-operator-optimizer/scripts/api_lookup.py TSCM TQueBind Matmul
python .agents/skills/ascend-operator-optimizer/scripts/api_lookup.py "Nd2NzParams" "DataCopy"
```

脚本会定位 API 快照目录，按文件名和正文命中排序，输出候选文档路径、标题和首个命中行。打开候选文档后再做具体判断。

没有 Python 或脚本不可用时，使用 `rg`：

```bash
rg --files reference/asc-devkit-api-9.0.0/docs/api | rg "DataCopyPad|TSCM|Matmul"
rg -n "Nd2NzParams|DataCopyExtParams|TSCM" reference/asc-devkit-api-9.0.0/docs/api/context
rg -n "GetLibApiWorkSpaceSize|Workspace|TCubeTiling" reference/asc-devkit-api-9.0.0/docs/api/context
```

## 场景路由

| 场景 | 首选查询词 | 常见候选文件 |
| --- | --- | --- |
| GM/UB 连续或非连续搬运 | `DataCopy`, `DataCopyPad`, `DataCopyExtParams`, `DataCopyPadExtParams` | `context/DataCopy.md`, `context/DataCopyPad(ISASI).md`, `context/DataCopy简介.md` |
| ND/NZ 随路转换 | `Nd2NzParams`, `随路转换ND2NZ搬运`, `随路转换NZ2ND搬运` | `context/随路转换ND2NZ搬运.md`, `context/随路转换NZ2ND搬运.md`, `context/tensor_api/arch/DataCopy.md` |
| 队列和资源生命周期 | `TQue`, `TQueBind`, `TSCM`, `TPipe`, `TBuf`, `AllocTensor`, `EnQue`, `DeQue`, `FreeTensor` | `context/TQue.md`, `context/TQueBind.md`, `context/TSCM.md`, `context/TPipe.md`, `context/TBuf.md` |
| Vector 基础计算 | `Cast`, `Add`, `Adds`, `Mul`, `Muls`, `Duplicate`, `Compare`, `BinaryRepeatParams` | `context/Cast.md`, `context/Add.md`, `context/Adds.md`, `context/Mul.md`, `context/Duplicate.md`, `context/Compare.md` |
| Reduce/归约 | `ReduceSum`, `ReduceMax`, `WholeReduceSum`, `BlockReduceSum`, `tmpBuffer` | `context/ReduceSum.md`, `context/WholeReduceSum.md`, `context/BlockReduceSum.md`, `context/归约计算.md` |
| 量化/反量化 | `AscendAntiQuant`, `AntiQuantize`, `AscendQuant`, `AscendDequant`, `Dequantize` | `context/AscendAntiQuant.md`, `context/AntiQuantize.md`, `context/AscendQuant.md`, `context/AscendDequant.md` |
| Cube/Matmul/MMAD | `Matmul`, `TCubeTiling`, `BatchMatmulGetTmpBufSize`, `Fixpipe`, `LoadData`, `TSCM` | `context/矩阵计算.md`, `context/TCubeTiling结构体.md`, `context/Fixpipe.md`, `context/tensor_api/arch/LoadData.md` |
| workspace | `GetLibApiWorkSpaceSize`, `Workspace`, `GetSysWorkSpacePtr`, `GetWorkspaceLen` | `context/GetLibApiWorkSpaceSize.md`, `context/Workspace.md`, `context/GetSysWorkSpacePtr.md` |
| 同步和事件 | `AllocEventID`, `FetchEventID`, `TQueSync`, `CrossCoreSetFlag`, `CrossCoreWaitFlag`, `SyncAll` | `context/AllocEventID.md`, `context/FetchEventID.md`, `context/TQueSync.md`, `context/同步控制.md` |
| 平台信息 | `GetCoreNum`, `GetCoreNumAiv`, `GetCoreNumAic`, `GetRuntimeUBSize`, `GetSocVersion` | `context/GetCoreNum.md`, `context/GetCoreNumAiv.md`, `context/GetRuntimeUBSize.md`, `context/GetSocVersion.md` |

## 查找步骤

1. 把问题转成 API 关键词：函数名、结构体名、参数名、中文主题名各准备一个。
2. 先搜文件名，优先打开同名 Markdown。
3. 同名文档不存在时搜正文，优先看包含函数签名、模板参数、输入输出、约束、示例的页面。
4. 对照目标代码确认：
   - 目标位置是 GM、UB、TSCM、L1 还是 L0。
   - 源和目的 tensor 类型是否匹配。
   - 参数单位是字节、元素个数还是 32B DataBlock。
   - `repeatTime`、blockCount、stride、queue depth 等是否有上限。
   - 是否有平台或 ISASI 兼容性说明。
5. 只把查到的具体事实写进设计、注释或报告。不要把未验证推断写成 API 结论。

## AWQ W4A16 相关提醒

- AIV 侧 ND2NZ 写 GM workspace 前必须查 `DataCopy` overload。一次失败经验是没有找到 `DataCopy(GlobalTensor, LocalTensor, Nd2NzParams)` 可用重载。
- TSCM 与 Matmul 高阶 API 联动前，先查 `TSCM.md`、`TQueBind.md`、`TPosition.md` 和 Matmul/TCubeTiling 相关页面；不要只凭示例调整 queue depth 或 position。
- `AscendAntiQuant`、`AntiQuantize` 等高阶 API 可能需要 tmp buffer。查文档时同时搜索 `GetAntiQuantizeTmpBufferFactorSize` 和 `GetAntiQuantizeMaxMinTmpSize`。
- workspace-aware wrapper 相关问题先查 `Workspace`、`GetLibApiWorkSpaceSize`，再核对目标工程生成 wrapper 的实际参数顺序。

## 记录格式

当 API 查询影响设计或修复时，在报告中记录：

```text
API evidence:
- Query: DataCopy Nd2NzParams
- Docs: reference/asc-devkit-api-9.0.0/docs/api/context/随路转换ND2NZ搬运.md
- Conclusion: <只写文档支持的事实>
- Validation: <编译/测试命令或失败日志>
```
