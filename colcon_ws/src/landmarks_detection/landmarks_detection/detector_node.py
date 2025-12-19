import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import torchvision.transforms as T
import cv2
import sys
from pathlib import Path

# -----------------------
# Deformable-DETR import
# -----------------------
DETR_ROOT = Path("/home/roboworks/Visual-Localization/Deformable-DETR")
sys.path.insert(0, str(DETR_ROOT))
print(sys.path)
from models import build_model
from util.misc import nested_tensor_from_tensor_list

# -----------------------

class DeformableDETRNode(Node):

    def __init__(self):
        super().__init__("deformable_detr_node")

        self.declare_parameter("checkpoint")
        self.declare_parameter("image_topic", "/camera/image_raw")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.bridge = CvBridge()

        self.model, self.postprocessors = self.load_model(
            self.get_parameter("checkpoint").value
        )

        topic = self.get_parameter("image_topic").value
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10
        )

    def load_model(self, checkpoint_path):
        self.get_logger().info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 🔑 USAR LOS ARGS DEL TRAINING
        args = checkpoint["args"]

        model, _, postprocessors = build_model(args)
        model.load_state_dict(checkpoint["model"], strict=False)

        model.to(self.device)
        model.eval()

        return model, postprocessors


    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        h, w, _ = img_rgb.shape
        orig_size = torch.tensor([[h, w]], device=self.device)

        results = self.postprocessors["bbox"](outputs, orig_size)[0]

        scores = results["scores"]
        keep = scores > 0.6

        boxes = results["boxes"][keep]

        # DEBUG VISUAL
        for box in boxes:
            x0, y0, x1, y1 = box.int().tolist()
            cv2.rectangle(cv_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.imshow("detections", cv_img)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = DeformableDETRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
