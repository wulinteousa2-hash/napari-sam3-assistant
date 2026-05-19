# Model Setup

SAM3 model files are downloaded separately from the Python package.

## Download Model Files

Download from Meta's gated Hugging Face repositories:

- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam3.1

You must request or accept access before the files are available. After access is
approved, download the files from the `Files and versions` tab.

![SAM3 model files screenshot](sam3_model_files.png)

## Model Folder Layout

The model files can be stored in any folder. The plugin only needs the folder
path.

Example SAM3.0 folder:

```text
D:\models\sam3\
  config.json
  processor_config.json
  sam3.pt
```

`model.safetensors` is also supported as a SAM3.0 weight file.

Example SAM3.1 folder:

```text
D:\models\sam3_1\
  config.json
  processor_config.json
  sam3.1_multiplex.pt
```

## Current Model Support

| Model files | Supported use in this plugin |
| --- | --- |
| SAM3.0 weights | 2D image tasks and 3D/video propagation |
| SAM3.1 `sam3.1_multiplex.pt` | 3D/video propagation through the SAM3.1 multiplex video predictor |

SAM3.1 is not currently routed through the plugin's 2D image model loader.

## Selecting the Model in napari

1. Open `Plugins > SAM3 Assistant`.
2. Click `Model Folder`.
3. Select the folder that contains `config.json`, `processor_config.json`, and
   the weight file.
4. In `Advanced` mode, choose `SAM3.0` or `SAM3.1` when needed.
5. Use `Validate` in Advanced mode if you want to check the selected folder
   before running inference.

## Device Rule

- Use `GPU / CUDA` for normal use.
- Use `GPU / CUDA` for SAM3.1 and 3D/video propagation.
- Use `CPU` only for experimental SAM3.0 2D workflows with a CPU-safe SAM3
  backend.

The top-level device control is environment-driven. Manual device override is
only exposed when backend testing is enabled with:

```bash
NAPARI_SAM3_ENABLE_DEVICE_OVERRIDE=1
```
