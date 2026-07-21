# Sources and Provenance

## Authority

- Active CANN 9.0.0 headers and generated compiler metadata
- Official CANN 9.0.0 Ascend C API and programming guides
- `cann/asc-devkit` branch/tag `9.0.0` examples
- Ascend `agent-skills` operator design, code generation, compile/debug, precision, performance, and MSSanitizer material
- on-device run records produced by this repository's AWQ optimization work

## Distillation Boundary

These references preserve decision rules, constraints, API search terms, and measured success/failure patterns. They do not copy the full CANN API documentation. Search local declarations and official same-version examples for signatures and product support.

The AWQ case file contains environment-specific measurements and rejected experiments. Treat it as evidence for candidate selection, not a universal hardware specification.

## Documentation Search

Search by exact symbol first, then by related position/format/product terms. For example:

```text
SetTensorB + LocalTensor + VECOUT
DataCopyPad + TSCM + ND2NZ
TSCM + full load + CubeFormat::NZ
Matmul + workspace + REGIST_MATMUL_OBJ
```

When a webpage and local headers disagree, record the CANN version and use the locally compilable 9.0.0 contract.
