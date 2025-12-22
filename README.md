# Visual-Localization
Monte Carlo Localization based on landmarks detection using **Deformable-DETR**.
---

## Prerequisites
It is recommended to use a virtual environment.

Example:
```
python3 -m venv <name_of_venv>
source <name_of_venv>/bin/activate

pip install "numpy<2" opencv-python==4.8.1.78 torch torchvision
```
Make sure that ROS 2 is using the same Python interpreter as your virtual environment.
If ROS 2 points to a different Python installation, you may encounter import errors even if all packages are correctly installed.

If the shebang points to a different Python version, edit the first line of the file:
```
Visual-Localization/colcon_ws/install/landmarks_detection/lib/landmarks_detection/detector
```
and replace it with:
```
#!/path/to/your/venv/bin/python
```
---
### Deformable-DETR setup
#### Clone the official repository
``` 
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR
```
#### Install Deformable-DETR Python dependencies
```
pip install -r requirements.txt --no-deps

pip install cython pycocotools timm
```
#### Deformable-DETR requires a CUDA-enabled PyTorch build.
Remove any previously installed PyTorch versions:
```
pip uninstall -y torch torchvision torchaudio
```
Install the correct CUDA build:
```
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```
Verify CUDA support:
```
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
>>true 11.8
```
**Note** Installing nvidia-cuda-toolkit is usually not required if your NVIDIA drivers are already installed. ONly install it if CUDA is missing on your system:
```
sudo apt install nvidia-cuda-toolkit
```
#### Verify installed versions
```
python - << 'EOF'
import numpy
import cv2
import torch
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
EOF
```
Expected output
```
NumPy: 1.26.4
OpenCV: 4.8.1
Torch: 2.0.1+cu118
CUDA: 11.8
```
⚠️ NumPy 2.x is not compatible with PyTorch / OpenCV / ROS2
If necessary, fix NumPy and OpenCV versions:
```
pip install numpy==1.26.4 

pip install opencv-python==4.8.1.78 
```
#### Compile Deformable-DETR CUDA ops
Inside the directory:
```
Deformable-DETR/models/ops
```
run: 
```
python setup.py build develop 
```
Verify successful compilation:
```
python - << 'EOF'
import MultiScaleDeformableAttention
print("MultiScaleDeformableAttention OK")
EOF
```
#### Patch Deformable-DETR for torchvision compatibility
Open:
```
Deformable-DETR/util/misc.py
``` 
Add the following import:
```
from packaging.version import Version
```
Then, approximately around lines **29** and **56**, modify the condition statements to the folowwing:
```
if Version(torchvision.__version__) < Version('0.5.0'):
...
elif Version(torchvision.__version__) < Version('0.7.0'):
...
```
### Common issues and fixes
#### C++ standard / CUDA compiler incompatibility
**Description**

This error appears during compilation with NVCC and originates from the C++ standard library (std_function.h). The compiler reports that template parameter packs are not expanded, even though the code itself is valid.

Example error:
```
/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’:
...
error: command '/usr/bin/nvcc' failed with exit code 1
```
**Solution**

Install GCC 10 and force its usage:
```
sudo apt install gcc-10 g++-10
export CC=gcc-10
export CXX=g++-10
```
Recompile:
```
cd ~/Deformable-DETR/models/ops
rm -rf build/
python setup.py build develop 
```

## Running the detection node
### Download the model
Download the trained model from the [`Files & Media`](https://www.notion.so/lira-pumanoids/2152da800dfb809d908be44c1834ace4?v=2152da800dfb808a9cce000cf964cb48&p=2d12da800dfb80fd822bea267912db9d&pm=s) section of the Organization Notion page.

### Set environment variables
DETR_ROOT must point to the Deformable-DETR repository:
```
export DETR_ROOT=<path_to_repo_Deformable-DETR>
```
### Build and run
```
cd ~/Visual-Localization/colcon_ws
colcon build
source install/setup.bash
ros2 run landmarks_detection detector \
  --ros-args \
  -p checkpoint:=<path_to_model.pth> \
  -p image_topic:=<image_topic>
```
