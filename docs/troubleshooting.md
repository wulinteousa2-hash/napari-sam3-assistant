# Troubleshooting

## No Mask Appears and Status Says `objects=0`

SAM3 ran but returned no detections above threshold.

Try:

- a shorter text prompt;
- a more common concept phrase;
- a lower `Detection threshold`;
- a box or exemplar prompt;
- a CUDA/PyTorch/SAM3 build compatible with your GPU.

## Text Prompt Creates No Prompt Layer

That is expected. Text segmentation does not need a prompt layer. Enter text and
click `Run Preview`.

## CUDA Kernel Image Error

Error:

```text
CUDA error: no kernel image is available for execution on the device
```

The GPU is visible, but at least one required CUDA kernel was not built for the
device architecture. Install compatible PyTorch, torchvision, and SAM3 builds
for the GPU.

For CPU-only 2D use, see [CPU-only SAM3.0 setup](cpu_only.md).

## Invalid GPU Architecture

Error:

```text
nvrtc: error: invalid value for --gpu-architecture
```

The installed PyTorch CUDA runtime cannot compile for the detected GPU. Install
a PyTorch, torchvision, and SAM3 build that supports the GPU.

## BFloat16 Conversion Errors

The plugin converts SAM3 `bfloat16` outputs to `float32` before writing
NumPy-backed napari layers. If dtype errors remain, restart napari after changing
device mode and run again.

## SAM3.1 `start_session` Fails With `unexpected keyword argument 'offload_state_to_cpu'`

This is a plugin/backend API mismatch.

- The failure happens during `start_session` or `init_state`.
- The installed `sam3` backend does not accept a keyword used by newer plugin
  code.
- The plugin includes compatibility handling for older installed backends.

If you still see the exact error, verify that napari is importing the intended
local `sam3` install and not an older duplicate environment copy.

## SAM3.1 Propagation Fails With `No available kernel. Aborting execution!` on Windows

This is different from the `offload_state_to_cpu` startup mismatch.

- `start_session` succeeds.
- prompts are accepted.
- failure happens when SAM3.1 multiplex propagation starts.

On some Windows systems using `triton-windows`, this appears to be an upstream
SAM3 runtime/kernel path issue. Use the documented workaround:

[windows_sam31_workaround/README.md](../windows_sam31_workaround/README.md)

## ARM64, CUDA, and DGX Spark

For ARM64 systems such as NVIDIA DGX Spark / GB10:

- Use Python 3.11 or newer.
- Keep the NVIDIA driver and CUDA stack current.
- Install a PyTorch/torchvision build that supports your GPU architecture.
- Use a PyTorch/torchvision/SAM3 build with compatible CUDA kernels.

GB10 reports compute capability `12.1` (`sm_121`). If your PyTorch build does
not include compatible kernels, you may see:

```text
CUDA error: no kernel image is available for execution on the device
nvrtc: error: invalid value for --gpu-architecture
```

The plugin does not compile PyTorch, torchvision, or SAM3 CUDA extensions.

## Check Which SAM3 Package napari Imports

Run this inside the same environment used to launch napari:

```bash
python - <<'PY'
import sam3
print(sam3.__file__)
PY
```

If the path points to a different environment or an old checkout, reinstall SAM3
in the active environment.

## Check PyTorch GPU Support

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list())
PY
```
