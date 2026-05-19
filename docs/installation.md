# Installation

SAM3 is not bundled with this plugin. Install the SAM3 backend and download the
SAM3 model files separately from Meta's Hugging Face repositories.

## Standard Windows Setup

1. Install Miniforge:

   https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe

2. Open Miniforge Prompt or PowerShell.

3. Create and activate an environment:

```bash
conda create -n napari-sam3 python=3.11 -y
conda activate napari-sam3
```

4. Install base Python tools and napari:

```bash
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"
python -m pip install "napari[all]"
```

5. Install CUDA-enabled PyTorch:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

6. Install SAM3:

```bash
python -m pip install --no-cache-dir git+https://github.com/facebookresearch/sam3
```

For a local SAM3 checkout instead:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
python -m pip install --no-cache-dir -e .
```

7. Install extra dependencies commonly needed by SAM3:

```bash
python -m pip install einops triton-windows pycocotools
```

8. Install the plugin:

```bash
python -m pip install napari-sam3-assistant
```

For a local plugin checkout:

```bash
python -m pip install -e .
```

9. Launch napari:

```bash
napari
```

If SAM3.1 multiplex propagation later fails on Windows with
`No available kernel. Aborting execution!`, see
[windows_sam31_workaround/README.md](../windows_sam31_workaround/README.md).

## Linux ARM64 / AArch64 Setup

Install Miniforge first:

```bash
chmod +x Miniforge3-Linux-aarch64.sh
./Miniforge3-Linux-aarch64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/bin/activate"
```

Create the environment:

```bash
conda create -n napari-sam3 python=3.11 -y
conda activate napari-sam3
```

Install napari:

```bash
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82" "numpy>=1.26,<2"
python -m pip install "napari[bermuda,pyqt6,optional-numba,optional-base]"
```

Install CUDA-enabled PyTorch:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Install SAM3:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
python -m pip install --no-cache-dir -e .
```

Install extra dependencies and the plugin:

```bash
python -m pip install einops triton pycocotools
python -m pip install napari-sam3-assistant
```

For a local plugin checkout:

```bash
python -m pip install -e .
```

Launch napari:

```bash
napari
```

## Verify the Environment

Check that SAM3 imports:

```bash
python -c "import sam3; print('sam3 import OK')"
```

Check napari:

```bash
python -c "import napari; print(napari.__version__)"
```

Check CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

## CPU-Only Option

CPU-only use is experimental and limited to SAM3.0 2D image workflows with a
CPU-safe SAM3 backend. Do not mix the standard Meta SAM3 package and a CPU fork
in the same environment because both provide an importable package named
`sam3`.

See [CPU-only SAM3.0 setup](cpu_only.md).
