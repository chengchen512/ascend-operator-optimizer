# Host and Tiling Code Rules

## Validation

- Null-check optional descriptors, tensors, outputs, workspace, and platform objects before use.
- Validate rank, shape relationships, dtype, layout, transpose flags, group sizes, axis/index ranges, and overflow.
- Use 64-bit arithmetic until a checked narrowing conversion is required by an API field.

## Platform

Query AIC/AIV counts, UB/L1/L0 resources, and library workspace through the active platform API. Select blockDim and TilingKey from workload and capability, not hardcoded target-shape identity.

## TilingData

- Keep fields minimal and deterministic.
- Preserve existing field order and historical meaning.
- Append new fields and use key/version selection for incompatible routes.
- Serialize exact valid lengths and aligned pitches when both are required.

## Matmul Tiling

Keep `MatmulType`, Host `SetAType/SetBType/SetCType`, formats, transpose flags, and Kernel `SetTensorA/B` source positions consistent. Cache expensive legality/tiling decisions when the framework invokes Host tiling repeatedly for the same shape.

## Workspace-Aware Launch

Expose the existing Python/public interface while requesting and passing required system/user workspace through the framework wrapper. Never hide required Matmul workspace behind a no-workspace launch.

## Output

Return clear errors for unsupported shapes or resources. A generic correct fallback is preferable to silent wrong tiling, but it must respect forbidden data paths and custom-operator constraints.
