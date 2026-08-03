# Compile and Debug

## Build Capture

Run the real project build after activating CANN 9.0.0. Save the exact command, compiler output, generated kernel metadata, and package/install result under the current `.ascendc-opt/runs/<run>/` directory.

## Compile Triage

1. Find the first source error, not the final wrapper failure.
2. Confirm the local header overload, template parameters, namespace, dtype, `TPosition`, and product guard.
3. Reduce uncertain code to a minimal compile probe.
4. Inspect generated metadata for blockDim, workspace, tiling key, and binary registration.

## Launch Triage

Use the target framework path. A direct kernel test does not prove ACLNN/Python wrapper behavior. For uncertain block semantics, run a coverage probe that records observed block indices and full output coverage.

## Runtime Triage

- Reproduce with synchronization at the test boundary so errors surface at the failing call.
- Search plog for `errorStr`, MPU, fixp, stream, and kernel names.
- Reduce to one core/tile only as a probe, then restore general execution.
- For hangs, distinguish host wait, device idle, event deadlock, and excessive runtime.

## Escalation

Use MSSanitizer for memory, initialization, race, or synchronization suspicion. A clean sanitizer run narrows evidence but does not prove numerical correctness.
