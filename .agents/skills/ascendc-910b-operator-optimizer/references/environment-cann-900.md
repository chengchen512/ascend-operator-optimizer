# CANN 9.0.0 Environment

## Authority Order

1. Active CANN 9.0.0 headers and compiler behavior.
2. Active device/runtime behavior and profiler output.
3. Official CANN 9.0.0 API documentation and `asc-devkit` 9.0.0 examples.
4. Distilled Skill references.

Do not use an API only because a newer, beta, or different-product document contains it.

## Inspect

Record these facts in `.ascendc-opt/environment.json`:

- `ASCEND_HOME_PATH` and resolved `set_env.sh`
- CANN package/version files
- detected SoC and logical/physical device mapping
- `bisheng`, `opc`, `msprof`, `npu-smi`, Python, PyTorch, and torch-npu versions
- AIC/AIV counts and UB/L1/L0/workspace values queried from the platform
- relevant environment variables and compiler options

Typical checks:

```bash
echo "$ASCEND_HOME_PATH"
find "$ASCEND_HOME_PATH" -maxdepth 3 -name set_env.sh -o -name version.info
command -v bisheng opc msprof npu-smi
npu-smi info
```

Activate the exact environment in every fresh build or run shell:

```bash
source /path/to/cann-9.0.0/set_env.sh
```

## Version Conflict

When documentation and local headers disagree:

1. Search the local declaration and all overloads.
2. Check product/SoC guards and template constraints.
3. Compile a minimal probe using the real compiler flags.
4. Run it on the target device when behavior is not compile-time decidable.
5. Record the result in the operator design or failure pattern.

## Environment Gate

Stop before code generation or optimization when CANN cannot be identified as 9.0.0, the target is not an Ascend 910B-family device, the compiler is unavailable, or the real framework launch path cannot execute a probe.
