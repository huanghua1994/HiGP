# macOS Installation Guide for HiGP

## Important Notice

This is a **temporary demo version** for Apple Silicon Macs. Official pip support is under development. Please be aware:
- Performance may be limited compared to the official release
- Unknown bugs may exist
- This build is for testing and development purposes only

## Prerequisites

### Required Hardware
- Apple Silicon Mac (M1/M2/M3/M4 series)
- MacOS 13.3 or newer

### Python Dependencies
- Python 3.10 or newer
- NumPy 2.2.2 or newer
- SciPy 1.15 or newer
- PyTorch 2.5.1 or newer
- Matplotlib (optional, for plotting)

```bash
pip install numpy scipy torch torchvision matplotlib pandas requests ipykernel ipywidgets
```

## Advanced Build Instructions

Since pip installation is not yet available for Apple Silicon, you need to build from source:

```bash
cd py-interface
python setup.py bdist_wheel
pip install dist/higp-*.whl
```