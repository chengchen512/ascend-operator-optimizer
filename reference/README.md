# Reference Skill Contents

本目录归档本次 AWQ W4A16 AscendC 算子设计、实现、远端编译测试和性能优化过程中实际使用到的 skill 内容，便于后续复盘和离线查阅。

## 目录结构

```text
reference/
├── README.md
├── asc-devkit-api-9.0.0/
│   ├── SOURCE.md
│   └── docs/api/
└── skills/
    ├── ascendc-operator-design/
    ├── ascendc-operator-performance-optim/
    ├── ascendc-operator-code-gen/
    ├── ascendc-operator-compile-debug/
    ├── ascendc-operator-project-init/
    ├── catlass-operator-design/
    └── ssh-remote-connect/
```

## 内容来源

内容来自本次 AWQ W4A16 工作区中实际读取和使用过的本地 skill 快照，不包含远端服务器上的临时实验文件。

| Skill | 已归档内容 | 本项目中的作用 |
|---|---|---|
| `ascendc-operator-design` | `SKILL.md`, `references/`, `templates/design-template.md` | 初始算子设计、接口、tiling 和设计文档模板。 |
| `ascendc-operator-performance-optim` | `SKILL.md`, `references/` | 性能排查闭环，覆盖 tiling、搬运、API、内存、流水。 |
| `ascendc-operator-code-gen` | `SKILL.md`, `references/`, `templates/` | AscendC host/kernel/CMake 修改参考，包括 DataCopy、Vector、Resource、Sync 和 Kernel 约束。 |
| `ascendc-operator-compile-debug` | `SKILL.md` | 远端 `build.sh`、wheel 安装、正确性测试流程。 |
| `ascendc-operator-project-init` | `SKILL.md`, `scripts/detect_ascend_kernel_project.sh`, `templates/ascend-kernel/` | 标准 `ascend-kernel` 工程骨架和初始化检查。 |
| `catlass-operator-design` | `SKILL.md`, `references/` | 早期评估 GEMM/CATLASS 路线是否适用。 |
| `ssh-remote-connect` | `SKILL.md`, 连接执行脚本和 `connection.env.example` | Windows/PowerShell 下通过 Paramiko 连接远端 Ascend 服务器并执行测试。 |

## 官方 API 文档快照

| 路径 | 来源 | 用途 |
|---|---|---|
| `reference/asc-devkit-api-9.0.0/docs/api/` | `https://gitcode.com/cann/asc-devkit/tree/9.0.0/docs/api` | 离线查阅 Ascend C API 入口、接口参数、约束和配图。 |

快照来源记录见 `reference/asc-devkit-api-9.0.0/SOURCE.md`。本次下载内容包含 2915 个文件，其中 Markdown 2090 个、PNG 823 个。

## 安全排除项

以下内容未归档：

- `.agents/skills/ssh-remote-connect/scripts/connection.local.env`
- `.agents/skills/ssh-remote-connect/scripts/set-password.ps1`
- `.agents/skills/ssh-remote-connect/scripts/set-password.sh`
- `__pycache__/`
- `*.pyc`
- 任何远端服务器密码、token、私钥或本地凭据

## 关联文档

- `knowledge/awq-skill-usage.md`: 本项目中 skill 的实际使用记录和关键测试结果。
- `knowledge/awq-w4a16-ascendc-experience.md`: 本次 AWQ W4A16 AscendC 算子优化经验总结。
- `.agents/skills/ascend-operator-optimizer/references/cases/awq-w4a16-ascendc.md`: 面向 optimizer skill 的压缩案例。
- `reference/asc-devkit-api-9.0.0/docs/api/README.md`: Ascend C API 官方文档快照入口。

## 使用建议

- 需要复现流程时，先读对应 skill 的 `SKILL.md`。
- 需要修改 AscendC kernel 时，优先查 `reference/skills/ascendc-operator-code-gen/references/`。
- 需要继续性能优化时，优先查 `reference/skills/ascendc-operator-performance-optim/references/`。
- 需要远端测试时，使用 `ssh-remote-connect` 的脚本说明，但不要在仓库中保存真实连接密码。
