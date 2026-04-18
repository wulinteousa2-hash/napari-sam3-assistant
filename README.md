# napari-sam3-assistant

`napari-sam3-assistant` is a napari widget for running Meta SAM 3 segmentation workflows from napari image, points, shapes, labels, and text inputs.

The plugin focuses on task-based segmentation workflows:

- 2D segmentation with text, box, point, and mask-style prompts
- 3D stack / video-like propagation from prompts on a selected slice or frame
- exemplar segmentation from Shapes ROI boxes
- text-based concept segmentation
- refinement with positive and negative point prompts

SAM 3 is not bundled with this plugin. Install the SAM 3 backend and download the SAM 3 model files separately from Meta's Hugging Face repository.

## Status

This project is under active development. The current widget supports the main napari interaction model, local SAM 3 model loading, prompt collection, background execution, and writing results back to napari layers.

Known hardware note: NVIDIA DGX Spark / GB10 ARM64 systems are new and may require NVIDIA or preview PyTorch builds for full CUDA kernel coverage. CPU mode is the safest fallback.

## Requirements

- Python `>=3.12`
- napari `>=0.5`
- SAM 3 Python package importable as `sam3`
- SAM 3 checkpoint directory containing:
  - `config.json`
  - `processor_config.json`
  - one of `sam3.pt`, `model.safetensors`, or `pytorch_model.bin`
- PyTorch and torchvision installed for your platform
https://huggingface.co/facebook/sam3/tree/main
GPU use requires a PyTorch/torchvision/SAM3 stack compiled for your GPU architecture. If CUDA kernels are not available for the device, select `CPU` in the widget.

## Installation

Create an environment with Python 3.12 or newer:

```bash
conda create -n napari-sam3 python=3.12
conda activate napari-sam3
```

Install PyTorch for your platform. Use the official PyTorch selector when possible:

```bash
pip install torch torchvision
```

For CUDA systems, install a CUDA-enabled PyTorch build that matches your driver and platform. See the PyTorch installation page for current commands.

Install SAM 3:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

Install this napari plugin:

```bash
git clone <YOUR_REPOSITORY_URL>
cd napari-sam3-assistant
pip install -e .
```

Run napari:

```bash
napari
```

Open the widget from:

```text
Plugins > SAM3 Assistant
```


## Getting SAM 3 Model Files

This plugin does not ship with SAM 3 weights or model configuration files. You must obtain them from the official Hugging Face SAM 3 repository:

```text
https://huggingface.co/facebook/sam3
Access requirement
```
The SAM 3 repository is gated.

Sign in to your Hugging Face account.
Open the facebook/sam3 model page.
Request access by filling in the required form.
Wait until access is approved.
After approval, download the model files and configuration files from the Files and versions tab.

A reference screenshot is provided here:

docs/sam3_model_files.png
Files to download

Keep the downloaded SAM 3 files together in a single local directory. At minimum, the plugin expects the model directory to contain:

config.json
processor_config.json
one weight file:
sam3.pt, or
model.safetensors, or
pytorch_model.bin

Depending on the model layout from Hugging Face, the directory may also include tokenizer and text-related files such as:

merges.txt
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.json
Recommended local layout

Choose one directory and keep all downloaded SAM 3 files there. For example:

~/Projects/napari/sam3/model/

or

~/models/sam3/

or another project-local path of your choice.

Then, in the plugin widget, select that directory as the model directory.

About sam3/model

In this project, sam3/model/ is a user-created local folder for storing downloaded SAM 3 model assets. It is not provided by the upstream SAM 3 repository by default.

If you intentionally created sam3/model/ to hold the downloaded weights and configuration files, that is an acceptable layout for this project. The important requirement is that all required SAM 3 files remain together in one directory and that you select that directory in the widget.




## ARM64, CUDA, and DGX Spark

For ARM64 systems such as NVIDIA DGX Spark / GB10:

- Use Python 3.12 or newer.
- Keep the NVIDIA driver and CUDA stack current. DGX Spark systems commonly report a CUDA 13.x capable driver.
- Verify PyTorch sees the GPU:

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

GB10 reports compute capability `12.1` (`sm_121`). Some PyTorch or torchvision wheels may not include all kernels for this architecture. Symptoms include:

```text
CUDA error: no kernel image is available for execution on the device
nvrtc: error: invalid value for --gpu-architecture
```

If this happens:

- select `CPU` in the plugin for reliable execution
- or install a PyTorch/torchvision/SAM3 build that includes the required GB10/SM121 CUDA kernels
- use explicit `CUDA` only when you want to test a GPU build on that machine

The plugin does not compile PyTorch, torchvision, or SAM3 CUDA extensions.

## Model Setup

Download or prepare a local SAM 3 checkpoint directory. The widget expects a directory, not just a single file. `~/models/sam3` is the recommended default location, but any readable directory can be used.

Expected layout:

```text
~/models/sam3/
  config.json
  processor_config.json
  sam3.pt
```

Other supported weight names:

```text
model.safetensors
pytorch_model.bin
```

In the widget:

1. Select the SAM 3 model directory.
2. Click `Validate`.
3. Choose `Device`:
   - `Auto`: use CUDA only when the plugin considers it compatible; otherwise CPU.
   - `CUDA`: force CUDA.
   - `CPU`: force CPU.
4. Optionally click `Load Image Model` or `Load 3D/Video Model`.

The widget can also lazy-load the correct model when `Run Preview` is clicked.

The selected model directory is remembered by the widget, so users do not need to browse to the same path every time napari starts.

## Basic Workflow

1. Open an image in napari.
2. Open `SAM3 Assistant`.
3. Select the image in `Napari Layers > Image`.
4. Select a task.
5. Create a prompt layer if the task needs one.
6. Click `Run Preview`.
7. Inspect `SAM3 preview labels`, `SAM3 preview masks`, or `SAM3 preview boxes`.
8. Click `Save Result as Labels` to keep the result.

Use `Clear Preview` to remove generated preview layers without deleting prompts or saved labels.

## Tasks

### Text Segmentation

Use text to segment all matching instances of a concept.

Workflow:

1. Set `Task` to `Text segmentation`.
2. Leave `Prompt tool` as `Text only`.
3. Enter a short phrase, for example:

```text
cell
nucleus
myelin
myelin sheath
```

4. Click `Run Preview`.

No prompt layer is needed for text segmentation. `Create Prompt Layer` is not required.

Text prompts usually work better as short noun phrases than instructions. Prefer `myelin sheath` over `segment all the myelin rings`.

If the result says `objects=0`, SAM 3 ran but did not return masks above threshold.

### 2D Segmentation With Boxes

Use boxes to identify the target object or concept.

Workflow:

1. Set `Task` to `2D segmentation`.
2. Set `Prompt tool` to `Box`.
3. Click `Create Prompt Layer`.
4. Draw one or more rectangles in the `SAM3 boxes` Shapes layer.
5. Click `Run Preview`.

The output appears in preview layers.

### Exemplar Segmentation

Use example ROIs to segment similar objects.

Workflow:

1. Set `Task` to `Exemplar segmentation`.
2. Set `Prompt tool` to `Box`.
3. Click `Create Prompt Layer`.
4. Draw boxes around one or more example objects.
5. Click `Run Preview`.

The local SAM 3 image API exposes visual exemplars through geometric box prompts. The plugin stores ROI metadata, but inference currently passes exemplar ROIs as SAM 3 visual box prompts.

### Refinement With Positive and Negative Points

Use points to correct a result.

Workflow:

1. Set `Task` to `Refinement`.
2. Set `Prompt tool` to `Points (+/-)`.
3. Click `Create Prompt Layer`.
4. Choose `Positive` and add points on regions to include.
5. Choose `Negative` and add points on regions to exclude.
6. Click `Run Preview`.

This is useful after a text, box, or exemplar preview is close but not correct.

### Labels Mask Prompt

Use a napari Labels layer as a mask-style prompt.

Workflow:

1. Set a task that supports mask prompts.
2. Set `Prompt tool` to `Labels mask`.
3. Click `Create Prompt Layer`.
4. Paint non-zero pixels in `SAM3 mask prompt`.
5. Click `Run Preview`.

### 3D Stack / Video Propagation

Treat a stack as video-like data and propagate a prompt through frames or slices.

Workflow:

1. Open a stack in napari.
2. Set `Task` to `3D/video propagation`.
3. Select the target frame or slice in napari.
4. Create a prompt layer and add prompts on that frame.
5. Choose propagation direction:
   - `both`
   - `forward`
   - `backward`
6. Click `Run Preview` or `Propagate Stack/Video`.

Preview output is written to:

```text
SAM3 propagated preview labels
```

Saved output is written to:

```text
SAM3 saved propagated labels
```

## Channel Axis

`Channel axis` tells the plugin which data axis is color/channel.

Default:

```text
-1
```

Use `-1` for grayscale images and normal RGB/RGBA images. The plugin auto-detects trailing RGB/RGBA axes of size `3` or `4`.

Examples:

```text
(H, W)          -> -1
(H, W, 3)      -> -1
(H, W, 4)      -> -1
(Z, H, W)      -> -1
(C, H, W)      -> 0
(Z, C, H, W)   -> 1
(T, C, H, W)   -> 1
(Z, H, W, C)   -> 3
```

Leave it at `-1` unless your image has an explicit multi-channel microscopy dimension.

## Output Layers

Preview layers:

```text
SAM3 preview labels
SAM3 preview masks
SAM3 preview boxes
SAM3 propagated preview labels
```

Saved layers:

```text
SAM3 saved labels
SAM3 saved propagated labels
```

Buttons:

- `Run Preview`: run the selected task.
- `Clear Preview`: remove generated preview layers only.
- `Save Result as Labels`: copy preview labels into saved labels.
- `Cancel`: stop a running worker.
- `Unload`: unload the SAM3 model from memory.

## Troubleshooting

### No mask appears and status says `objects=0`

SAM 3 returned no detections above threshold. Try:

- a shorter text prompt
- a more common concept phrase
- box or exemplar prompts
- CPU mode if the CUDA path is unstable

### CUDA kernel image error

Error:

```text
CUDA error: no kernel image is available for execution on the device
```

The GPU is visible, but at least one required CUDA kernel was not built for the device architecture. Use `CPU`, or install compatible PyTorch/torchvision/SAM3 builds.

### Invalid GPU architecture

Error:

```text
nvrtc: error: invalid value for --gpu-architecture
```

The installed PyTorch CUDA runtime cannot compile for the detected GPU. Use `CPU` or install a build that supports the GPU.

### BFloat16 conversion errors

The plugin converts SAM3 `bfloat16` outputs to `float32` before writing NumPy-backed napari layers. If you still see dtype errors, restart napari after changing device mode and run again.

### Text prompt creates no layer

That is expected. Text segmentation does not need a prompt layer. Enter text and click `Run Preview`.

## Development

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

The test suite covers coordinate mapping, prompt collection, adapter utility behavior, and static widget UI checks. It does not download SAM 3 weights.

## References

- SAM 3 repository: https://github.com/facebookresearch/sam3
- PyTorch installation selector: https://pytorch.org/get-started/locally/
- napari plugin documentation: https://napari.org/plugins/
- napari publishing guide: https://napari.org/stable/plugins/testing_and_publishing/deploy.html

## License

MIT. See the project license file.
