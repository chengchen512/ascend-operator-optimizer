# API Selection and Lookup

## Lookup Order

1. Search local CANN 9.0.0 headers for the exact symbol and overload.
2. Search the local `asc-devkit` 9.0.0 checkout for a same-product example.
3. Search curated reference docs by API name, `TPosition`, format, dtype, and product constraint.
4. Compile a minimal probe before integrating an uncertain overload into a large kernel.

Typical commands:

```bash
rg -n "SetTensorB|DataCopyPad|TSCM|Mmad" "$ASCEND_HOME_PATH"/include
rg -n "SetTensorB|DataCopyPad|TSCM|Mmad" /path/to/asc-devkit/examples
rg -n "TPosition::VECOUT|CubeFormat::NZ|GetTiling" references
```

If `rg` is unavailable, use `grep -RIn` on the same narrow directories. Search declarations before prose.

## Selection Record

For every non-trivial API choice, record:

- exact overload and header
- source/destination `TPosition`
- dtype, layout/format, transpose, and alignment constraints
- count/repeat/stride units and numeric limits
- asynchronous behavior and required queue/event lifecycle
- product/CANN support
- required temporary memory or workspace

## Core Choices

- Use `TQue` for producer/consumer synchronization and `TBuf` for local scratch without queue semantics.
- Use `TQueBind`/`TSCM` only when the documented source, destination, lifecycle, event, and format constraints match.
- Prefer `DataCopyPad` for non-trivial GM/local copies; verify every unit.
- Use high-level Matmul when its call granularity and internal movement fit the operator. Evaluate Mmad/lower-level routes only with a complete L0/L1 and synchronization design.
- Use `SetTensorA/B(LocalTensor)` only with a supported `TPosition`, enough storage for the configured single tile, and the format expected by Matmul.

## High-Risk Searches

For local-B/TSCM/Cube work, search these together:

```text
MatmulType SetBType SetTensorB LocalTensor VECOUT TSCM A1 B1 B2 NZ ND
DataCopyPad ISASI nd2nz TQueBind AllocTensor EnQue DeQue FreeTensor
```

Do not infer that two individually supported API operations form a supported cross-engine data path. Prove the complete path with a repeatable probe.
