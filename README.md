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

This project is under active development. The current widget supports local SAM 3 model loading, napari prompt collection, background execution, and writing results back to napari layers.

## Requirements

- Python `>=3.12`
- napari `>=0.5`
- SAM 3 Python package importable as `sam3`
- PyTorch and torchvision installed for your platform
- A local SAM 3 checkpoint directory containing:
  - `config.json`
  - `processor_config.json`
  - one weight file such as `sam3.pt` or `model.safetensors`
GPU use requires a PyTorch / torchvision / SAM 3 stack that is compatible with your GPU architecture. If CUDA kernels are not available for the device, select **CPU** in the widget.

## Setup

Setup has three parts:

1. Install the SAM 3 backend.
2. Download the SAM 3 model files from Hugging Face and configure the model path.
3. Install this napari plugin.

### 1. Install SAM 3

Create and activate an environment:

```bash
conda create -n napari-sam3 python=3.12
conda activate napari-sam3
```

Install PyTorch and torchvision for your platform. Use the official PyTorch selector for the current command:

```bash
pip install torch torchvision
```

Install SAM 3:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### 2. Download SAM 3 Weights

This plugin does not ship with SAM 3 weights or model configuration files. Download them from the official Hugging Face repository:

```text
https://huggingface.co/facebook/sam3
```

The repository is gated. Sign in to Hugging Face, open `facebook/sam3`, accept or request access, then download the files from the repository.

Recommended download method:

```bash
pip install -U huggingface_hub
hf auth login
mkdir -p ~/models/sam3
hf download facebook/sam3 --local-dir ~/models/sam3
```

Expected model directory:

```text
~/models/sam3/
  config.json
  processor_config.json
  sam3.pt
```

`model.safetensors` is also supported as a weight file. Depending on the Hugging Face layout, the directory may also contain tokenizer files such as `tokenizer.json`, `vocab.json`, and `merges.txt`.

Keep all downloaded model files together in one directory. In the plugin widget, click `Browse`, select that directory, then click `Validate`.

Do not put downloaded weights inside the SAM 3 Python source package, for example `sam3/sam3/model`. If you want a project-local model folder, use a separate directory such as:

```text
~/Projects/napari/sam3/model
```

The selected model directory is remembered by the widget.

### 3. Install napari-sam3-assistant

Install this plugin:

```bash
git clone https://github.com/wulinteousa2-hash/napari-sam3-assistant
cd napari-sam3-assistant
pip install -e .
```

Start napari:

```bash
napari
```

Open the widget from:

```text
Plugins > SAM3 Assistant
```

## Basic Workflow

1. Open an image in napari.
2. Open `Plugins > SAM3 Assistant`.
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

The current SAM 3 video predictor backend is CUDA-only. CPU mode is supported for 2D/image workflows, not 3D/video propagation.

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

- `Validate`: check the selected SAM 3 model directory.
- `Load Image Model`: load the 2D/image model.
- `Load 3D/Video Model`: load the video propagation model.
- `Run Preview`: run the selected task.
- `Clear Preview`: remove generated preview layers only.
- `Save Result as Labels`: copy preview labels into saved labels.
- `Cancel`: stop a running worker.
- `Unload`: unload the SAM3 model from memory.

## ARM64, CUDA, and DGX Spark

For ARM64 systems such as NVIDIA DGX Spark / GB10:

- Use Python 3.12 or newer.
- Keep the NVIDIA driver and CUDA stack current.
- Install a PyTorch/torchvision build that supports your GPU architecture.
- Use `CPU` mode for reliable 2D execution if CUDA kernels are unavailable.
- Use explicit `CUDA` only when testing a compatible GPU build.

Check PyTorch GPU support:

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

GB10 reports compute capability `12.1` (`sm_121`). If your PyTorch build does not include compatible kernels, you may see:

```text
CUDA error: no kernel image is available for execution on the device
nvrtc: error: invalid value for --gpu-architecture
```

The plugin does not compile PyTorch, torchvision, or SAM 3 CUDA extensions.

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

The GPU is visible, but at least one required CUDA kernel was not built for the device architecture. Use `CPU`, or install compatible PyTorch/torchvision/SAM 3 builds.

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
- SAM 3 model files: https://huggingface.co/facebook/sam3
- PyTorch installation selector: https://pytorch.org/get-started/locally/

## License

MIT. See the project license file.
