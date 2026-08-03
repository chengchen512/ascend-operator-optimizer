# Sources and Provenance

## Authority

- Active CANN 9.0.0 headers and generated compiler metadata
- Official CANN 9.0.0 Ascend C API and programming guides
- `cann/asc-devkit` branch/tag `9.0.0` examples
- Ascend `agent-skills` operator design, code generation, compile/debug, precision, performance, and MSSanitizer material
- on-device run records produced by this repository's AWQ optimization work
- user-supplied 2026 Huawei CANN Open Software License `mmad.asc` example covering direct `Mmad` for int4, int8, bfloat16, float, bias modes, layout conversion, and Fixpipe

## Distillation Boundary

These references preserve decision rules, constraints, API search terms, and measured success/failure patterns. They do not copy the full CANN API documentation. Search local declarations and official same-version examples for signatures and product support.

The AWQ case file contains environment-specific measurements and rejected experiments. Treat it as evidence for candidate selection, not a universal hardware specification.

The supplied `mmad.asc` example spans multiple architectures. Preserve only the A2/A3-family decision rules that are confirmed by active 910B/CANN 9.0.0 headers or probes; do not transplant `__NPU_ARCH__ == 3510` branches into the 910B path.

## Documentation Search

Search by exact symbol first, then by related position/format/product terms. For example:

```text
SetTensorB + LocalTensor + VECOUT
DataCopyPad + TSCM + ND2NZ
TSCM + full load + CubeFormat::NZ
Matmul + workspace + REGIST_MATMUL_OBJ
```

When a webpage and local headers disagree, record the CANN version and use the locally compilable 9.0.0 contract.
