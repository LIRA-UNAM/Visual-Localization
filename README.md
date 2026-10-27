# Visual-Localization

Monte Carlo Localization (MCL) based on landmark detection with **Deformable-DETR**, implemented as ROS 2 packages.

| Package | Executables | Description |
|---|---|---|
| `landmarks_detection` | `detector` | Detects field landmarks with Deformable-DETR and publishes them on `/vision/landmarks` |
| `particle_filter` | `mcl_node`, `map_node` | Monte Carlo Localization and field map |
| `vision_msgs_def` | — | Custom messages (`VisionLandmark`, `VisionLandmarkArray`) |
| `booster_t1_webots` | `odom_publisher` | Webots simulation of the Booster T1 robot |

---

## 1. Requirements

| Component | Tested with | Notes |
|---|---|---|
| OS | Ubuntu 22.04 | |
| ROS 2 | Humble | Uses the system Python (**3.10**). Your venv must use the same Python version. |
| NVIDIA GPU + driver | — | **Required.** The Deformable-DETR attention operator is a custom CUDA kernel; there is no CPU path. |
| CUDA Toolkit (`nvcc`) | 11.8 | Must match the CUDA version of your PyTorch build (see §3). |
| PyTorch / torchvision | 2.0.1 / 0.15.2 | Any CUDA build that matches your driver and `nvcc` should work. |
| NumPy | 1.26.x | **NumPy 2.x is not compatible** with ROS 2 Humble (`cv_bridge`) and older OpenCV/PyTorch builds. |
| OpenCV | 4.8.1 | |

> The versions in "Tested with" are the ones used during development. They are not mandatory: what matters is that the **driver, `nvcc` and PyTorch agree on the CUDA version** (§3).

---

## 2. Python environment

Create the virtual environment **with access to the ROS 2 system packages** (`rclpy`, `cv_bridge`, etc. are installed system-wide, not via pip):

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages ~/venvs/vloc
source ~/venvs/vloc/bin/activate
touch ~/venvs/vloc/COLCON_IGNORE      # only if the venv is inside the workspace

pip install "numpy<2" opencv-python==4.8.1.78
```

Do **not** install `torch` yet; it depends on your CUDA version (next section).

---

## 3. Choosing your CUDA / PyTorch combination

Three things must be compatible:

1. **Driver** – `nvidia-smi` shows, top right, the *maximum* CUDA version the driver supports (e.g. `CUDA Version: 12.4`).
2. **PyTorch build** – the `cuXXX` suffix of the wheel (`cu118`, `cu121`, `cu124`, …). Must be ≤ the driver's CUDA version.
3. **CUDA Toolkit (`nvcc`)** – used to compile the Deformable-DETR operator. Its version must match `torch.version.cuda` (at least the same major version, ideally the same minor).

Check what you have:

```bash
nvidia-smi                 # driver and max supported CUDA
nvcc --version             # toolkit actually installed (may be missing)
```

Then pick the matching wheel from the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/). Examples:

```bash
# CUDA 11.8 (tested configuration)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

If a different `torch` was already installed, remove it first:

```bash
pip uninstall -y torch torchvision torchaudio
```

> **RTX 50xx (Blackwell, `sm_120`)** needs CUDA ≥ 12.8 and PyTorch ≥ 2.7 (`cu128`). Older builds will install but fail at runtime with *"no kernel image is available"*.

### Installing a matching CUDA Toolkit

The Deformable-DETR operator **is compiled with `nvcc`, so the toolkit is required** (having the driver is not enough).

- Preferred: install the exact version from the [NVIDIA CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive) (choose "runfile" or "deb (network)" and install only the toolkit, not the driver). It installs into `/usr/local/cuda-X.Y`.
- `sudo apt install nvidia-cuda-toolkit` installs whatever version Ubuntu ships (11.5 on 22.04), which usually **does not match** your PyTorch build. Avoid it unless it happens to match.

Point the build to the right toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-11.8     # adjust to your version
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Verify

```bash
python - << 'EOF'
import numpy, cv2, torch
print("NumPy :", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda, "| available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF
nvcc --version | tail -n 2
```

`available` must be `True`, NumPy must be `1.x`, and the `nvcc` release must match `torch.version.cuda`.

---

## 4. Deformable-DETR setup

### 4.1 Clone and install dependencies

```bash
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR
pip install pycocotools tqdm cython scipy timm
```

(`requirements.txt` is not installed directly because it would pull versions that conflict with the ones above.)

### 4.2 Patch `util/misc.py` (torchvision ≥ 0.10)

The original code compares versions with `float(torchvision.__version__[:3])`, which turns `0.15.2` into `0.1` and selects the wrong code path, causing import errors. Edit `Deformable-DETR/util/misc.py`:

```python
from packaging.version import Version   # add with the other imports
```

and replace the two comparisons (around lines 29 and 56):

```python
# before
if float(torchvision.__version__[:3]) < 0.5:
...
if float(torchvision.__version__[:3]) < 0.7:

# after
if Version(torchvision.__version__) < Version('0.5.0'):
...
if Version(torchvision.__version__) < Version('0.7.0'):
```

### 4.3 Compile the CUDA operator

```bash
cd Deformable-DETR/models/ops
rm -rf build/
# Optional: compile only for your GPU (faster, avoids unsupported-arch errors).
# Get the value from torch.cuda.get_device_capability(), e.g. (8, 6) -> "8.6"
export TORCH_CUDA_ARCH_LIST="8.6"
python setup.py build develop
```

Verify:

```bash
python -c "import torch, MultiScaleDeformableAttention; print('MultiScaleDeformableAttention OK')"
```

(`torch` must be imported first, otherwise you may get an `undefined symbol` error even when the build succeeded.)

---

## 5. Build the ROS 2 workspace

Build **with the venv activated**, so the generated executables use the venv's Python:

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/vloc/bin/activate
cd <path_to>/Visual-Localization/colcon_ws
rm -rf build install log          # if it was previously built with another Python
python -m colcon build --symlink-install
source install/setup.bash
```

Check the shebang of the generated script:

```bash
head -n 1 install/landmarks_detection/lib/landmarks_detection/detector
```

It should point to `~/venvs/vloc/bin/python`. If it says `/usr/bin/python3`, the build did not use the venv. Editing this line by hand works as a quick fix, but it is **overwritten on every `colcon build`**.

---

## 6. Running

### 6.1 Download the model

Download the trained checkpoint (`.pth`) from the [`Files & Media`](https://www.notion.so/lira-pumanoids/2152da800dfb809d908be44c1834ace4?v=2152da800dfb808a9cce000cf964cb48&p=2d12da800dfb80fd822bea267912db9d&pm=s) section of the organization's Notion page.

### 6.2 Environment (every new terminal)

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/vloc/bin/activate
source <path_to>/Visual-Localization/colcon_ws/install/setup.bash
export DETR_ROOT=<path_to>/Deformable-DETR
```

### 6.3 Landmark detector

```bash
ros2 run landmarks_detection detector --ros-args \
  -p checkpoint:=<path_to_model.pth> \
  -p image_topic:=/camera/color/image_raw \
  -p visualize:=true
```

| Parameter | Default | Description |
|---|---|---|
| `checkpoint` | — (required) | Path to the `.pth` model |
| `image_topic` | `/camera/color/image_raw` | Input image topic |
| `visualize` | `true` | Publish the annotated image |

| Topic | Type | Direction |
|---|---|---|
| `<image_topic>` | `sensor_msgs/Image` | subscribed |
| `/vision/landmarks` | `vision_msgs_def/VisionLandmarkArray` | published |
| `/vision/image_detections_pub` | `sensor_msgs/Image` | published (debug) |

### 6.4 Localization

```bash
ros2 run particle_filter map_node
ros2 run particle_filter mcl_node
```

---

## 7. Troubleshooting

### `nvcc` fails with `std_function.h ... parameter packs not expanded with '...'`

Known incompatibility between CUDA 11.x and GCC 11 (the default on Ubuntu 22.04). Use an older host compiler:

```bash
sudo apt install gcc-10 g++-10
export CC=gcc-10 CXX=g++-10 CUDAHOSTCXX=g++-10
cd Deformable-DETR/models/ops && rm -rf build/ && python setup.py build develop
```

With CUDA ≥ 12.x this is normally not needed. Each CUDA version supports a maximum GCC version; check the *Host compiler support* table in the CUDA installation guide for your version.

### `The detected CUDA version (X) mismatches the version that was used to compile PyTorch (Y)`

`nvcc` and `torch.version.cuda` differ. Install the matching toolkit, or install the PyTorch wheel that matches your `nvcc` (§3), and set `CUDA_HOME`.

### `CUDA is not available` / `NotImplementedError: Cuda is not available` during the build

`torch.cuda.is_available()` is `False`: you installed a CPU-only wheel, or the wheel's CUDA is newer than your driver supports. Reinstall from the correct `--index-url`.

### `no kernel image is available for execution on the device`

The operator or PyTorch was not compiled for your GPU architecture. Set `TORCH_CUDA_ARCH_LIST` and recompile (§4.3); for very new GPUs use a newer PyTorch/CUDA.

### `ModuleNotFoundError` (`torch`, `MultiScaleDeformableAttention`, `rclpy`, …) when running the node

- The executable is not using the venv: check the shebang (§5).
- The venv was created without `--system-site-packages`, so `rclpy` is not visible.
- `DETR_ROOT` is not exported in the current terminal.

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

```bash
pip install "numpy<2"
```