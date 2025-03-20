# MRT

### Environment Setup

```bash
conda create -n mrt_ntire python=3.9 -y
conda activate mrt_ntire
```

```bash
# ===== For GPU inference.
# Conda
conda install -c conda-forge numpy opencv onnx onnxruntime-gpu -y
# Pip
pip install numpy opencv-python onnx onnxruntime-gpu gdown

# ===== For CPU inference.
# Conda
conda install -c conda-forge numpy opencv onnx onnxruntime -y
# Pip
pip install numpy opencv-python onnx onnxruntime gdown
```

### Dataset setup
Please check details under ```datasets/README.md```.

### Inference
```bash
# NTIRE Low Light Image Enhancement
python test.py --comp LLIE

# NTIRE Image Shadow Removal
python test.py --comp ShadowR
```

### Contact
For any issues, please email ```albrateanu@gmail.com```.


