# YOLO Auto Labeler

A lightweight **YOLOv8-based automatic labeling tool** for image datasets.

This tool helps you quickly bootstrap annotations by:
- Loading a folder of images
- Automatically generating bounding boxes and class labels using a trained YOLOv8 model
- Browsing images in small batches for visual inspection
- Exporting annotations in **LabelMe-compatible JSON format**

The main goal is to **speed up dataset annotation**, while leaving fine-grained corrections and manual editing to tools such as **LabelMe**.

---
## Features

- 📂 Folder-based image loading  
- 🏷️ Automatic bounding box generation using YOLOv8  
- 🖼️ Batch visualization (multiple images per page)  
- 📄 LabelMe-compatible JSON output  
- ⚡ One-click full-folder auto-labeling  


---
## Requirements

- Python **3.9+**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Gradio

It is **strongly recommended** to use a virtual environment where YOLOv8 dependencies are installed.

---
### Install dependencies

```bash
pip install -r requirements.txt
```
---
## Running the Auto Labeler
Run the app in terminal:
```bash
python app.py \
  --model path_to_models/best.pt \
  --images path_to_dataset/images \
  --output outputs
```
After launching, open the URL printed in the terminal. It should look similar to:
```bash  
Running on local URL:  http://127.0.0.1:7860
```
Open this address in your web browser to access the interface.

---
## Output format
For each image, a .json file is generated in the output directory using a LabelMe-compatible format, making it easy to:

- Open and refine annotations in LabelMe

- Integrate the dataset into existing computer vision pipelines

---
## Notes

- This tool is intended as a dataset preparation utility, not a replacement for manual annotation tools.

- For precise label correction, class editing, or polygon refinement, use LabelMe after auto-labeling.