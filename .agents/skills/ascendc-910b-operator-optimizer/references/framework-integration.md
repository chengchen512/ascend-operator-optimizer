# Framework Integration

## Launch Profiles

| Profile | Typical path | Required proof |
| --- | --- | --- |
| `ascendc-standard` | project kernel through framework wrapper | full block coverage and workspace propagation |
| `ascendc-msopgen-aclnn-dynamic` | msopgen/aclnn custom op | real-path launch coverage probe |
| `ascendc-vector` | AIV/UB path | AIV block mapping and queue lifecycle |
| `ascendc-cube-matmul` | AIC/Matmul or MIX | Matmul workspace, tiling consistency, local format/lifetime |

## Interface Preservation

Keep the public Python/ACLNN signature stable unless the user requests an API change. Host autotiling, workspace allocation, TilingKey selection, and internal candidate routes belong below that interface.

## Workspace

Use a workspace-aware Host wrapper when Matmul/library or operator state requires it. Query and allocate system/user workspace in Host code, pass it through the launch layer, and bind the Matmul object to the provided workspace. Test zero and non-zero workspace routes.

## Launch Coverage Probe

The probe must use the real wrapper and record requested blockDim, observed indices/count, and output coverage. Failure blocks kernel optimization because ordinary `GetBlockIdx/GetBlockNum` assumptions are not established.

## Registration

Verify operator schema, dtype/layout inference, tiling registration, kernel name, binary metadata, shared library/package installation, and Python binding all point to the same implementation and workspace contract.
