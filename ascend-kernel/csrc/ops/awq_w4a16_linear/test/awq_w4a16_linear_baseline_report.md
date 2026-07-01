# awq_w4a16_linear baseline report

Baseline is the pure custom pack8 scalar accumulation route before vector dequant.

| Shape | Correctness | Candidate us | Vendor us | Ratio |
|---|---|---:|---:|---:|
| `[2,256,256]` | pass | 1935.0799560546875 | 8.511000126600266 | 227.36222855957809 |
| `[16,4096,22016]` | pass | 512605.5501302083 | 89.95999892552693 | 5698.149802720285 |