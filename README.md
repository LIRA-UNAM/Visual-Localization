# Visual-Localization
Monte Carlo Localization based on landmarks detection using Deformable-DETR.
## Prerequisites
It is recommended to use a virtual environment.

Example:
```
python3 -m venv <name_for_venv>
source venv/bin/activate

pip install numpy<2 opencv-python==4.8.1 torch torchvision
```
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
Installing nvidia-cuda-toolkit is usually not required if your NVIDIA drivers are already installed. ONly install it if CUDA is missing on your system:
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
If it is necessary, Fix NumPy and OpenCV versions.
```
pip install numpy==1.26.4 

pip install opencv-python==4.8.1.78 
```
#### Compile Deformable-DETR CUDA ops
Inside `Deformable-DETR/models/ops`directory: 
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
## How to run the detection node
### Download the model
Go to the [`Organization Notion link`](https://www.notion.so/lira-pumanoids/2152da800dfb809d908be44c1834ace4?v=2152da800dfb808a9cce000cf964cb48&p=2d12da800dfb80fd822bea267912db9d&pm=s) in Files&Media section.

``ros2 run landmarks_detection detector   --ros-args   -p checkpoint:= <path_to_model.pth>   -p image_topic:= <image_topic/>
``