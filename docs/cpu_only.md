# CPU-only SAM3.0 setup

CPU-only use requires a SAM3 backend that is safe on CPU. The standard Meta
`facebookresearch/sam3` package may still allocate CUDA tensors while building the
image model, even when the plugin passes `device="cpu"`.

The tested CPU-only path uses the unofficial `rhubarb-ai/sam3-cpu` fork as the
environment's importable `sam3` package.

## Supported in this plugin

- SAM3.0 2D image workflows
- point prompts
- box prompts
- exemplar prompts
- text prompts
- Live Points

Not supported in CPU mode:

- SAM3.1 multiplex
- 3D/video propagation
- workflows that use the SAM3 video predictor

GPU/CUDA is still recommended for full SAM3 functionality.

## Clean CPU environment

Use a separate environment from your CUDA install.

```bash
conda create -n napari-sam3-cpu python=3.12 -y
conda activate napari-sam3-cpu

python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"
python -m pip install "napari[all]"

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir git+https://github.com/rhubarb-ai/sam3-cpu
python -m pip install napari-sam3-assistant
```

For a local plugin checkout, replace the last command with:

```bash
python -m pip install -e .
```

Do not install Meta `facebookresearch/sam3` in this CPU environment. Both
packages provide an importable package named `sam3`, so mixing them can make it
unclear which backend napari is using.

## Verify the backend

Run this inside the activated CPU environment:

```bash
python - <<'PY'
import torch
import sam3
import sam3.model_builder

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("sam3 package:", sam3.__file__)
print("model_builder:", sam3.model_builder.__file__)
PY
```

Expected CPU-only PyTorch output:

```text
torch.version.cuda: None
torch.cuda.is_available(): False
```

The `sam3 package` and `model_builder` paths should point to the `sam3-cpu`
installation or your local clone of that fork.

## Model folder

Use a SAM3.0 model folder, not a SAM3.1 multiplex-only folder.

The folder should contain a SAM3.0 weight file such as:

```text
sam3.pt
```

The CPU backend may also need the BPE tokenizer file:

```text
bpe_simple_vocab_16e6.txt.gz
```

If your model folder has `merges.txt` instead, the plugin will try to create the
gzip file automatically:

```text
bpe_simple_vocab_16e6.txt.gz
```

You can also create it manually:

```bash
gzip -c merges.txt > bpe_simple_vocab_16e6.txt.gz
```

## Use in napari

1. Launch napari from the CPU environment.
2. Open `Plugins > SAM3 Assistant`.
3. Select the SAM3.0 model folder.
4. Set `Device` to `CPU`.
5. Use 2D image tasks such as points, boxes, text, exemplar, or Live Points.

If model loading fails with a message about CUDA tensors during image model
construction, napari is probably importing a non-CPU-safe SAM3 backend. Re-run
the backend verification command above.

## ARM64 / DGX Spark note

On some ARM64 systems, `decord` may not be available. If importing `sam3-cpu`
fails because `decord` is hard-imported, patch the local `sam3-cpu` checkout so
the import is optional in:

```text
sam3/train/data/sam3_image_dataset.py
```

The hard import:

```python
from decord import cpu, VideoReader
```

should be guarded so image-only CPU workflows can import without `decord`.
