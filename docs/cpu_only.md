# CPU-only SAM3.0 setup

CPU-only use requires a SAM3 backend that is safe on CPU. The standard Meta
`facebookresearch/sam3` package may still allocate CUDA tensors while building the
image model, even when the plugin passes `device="cpu"`.

The tested CPU-only path uses the external unofficial `rhubarb-ai/sam3-cpu` fork
as the environment's importable `sam3` package. The tested fork reports version
`0.1.0` and is distributed under the MIT license in its own repository.
`sam3-cpu` is not bundled with `napari-sam3-assistant`; install it separately in
the CPU-only environment.

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

# Required by sam3-cpu import path on Windows
python -m pip install decord

```
```bash
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
python -c "import torch; import sam3; import sam3.model_builder; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available():', torch.cuda.is_available()); print('sam3 package:', sam3.__file__); print('model_builder:', sam3.model_builder.__file__)"
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

Most users do not need to create this file manually. If automatic creation fails
because the model folder is read-only, run this from inside the model folder:

```bash
python -c "import gzip, shutil; shutil.copyfileobj(open('merges.txt','rb'), gzip.open('bpe_simple_vocab_16e6.txt.gz','wb'))"
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

On some Linux ARM64/aarch64 systems, including DGX Spark, `decord` may not provide a compatible prebuilt wheel. This can prevent `sam3-cpu` from importing, even when you only want to run image-only CPU workflows.

The typical error is:

```text
ModuleNotFoundError: No module named 'decord'
```
or an installation error showing that no matching decord wheel is available for Linux ARM64/aarch64.

This happens because sam3-cpu may hard-import decord in:

```text
sam3/train/data/sam3_image_dataset.py
```
For image-only CPU use, decord should not be required unless video decoding is actually used.

## Temporary workaround

Patch the local sam3-cpu checkout to make the decord import optional:


```bash
cd ~/Projects/napari/sam3-cpu

python - <<'PY'
from pathlib import Path

p = Path("sam3/train/data/sam3_image_dataset.py")
text = p.read_text()

old = "from decord import cpu, VideoReader"
new = """try:
    from decord import cpu, VideoReader
except ModuleNotFoundError:
    cpu = None
    VideoReader = None
"""

if old not in text:
    print("Target import line not found. File may already be patched.")
else:
    p.write_text(text.replace(old, new))
    print("Patched optional decord import in", p)
PY
```

Then verify that sam3-cpu imports correctly:
```bash
python -c "import sam3; print('sam3 import ok')"
```
