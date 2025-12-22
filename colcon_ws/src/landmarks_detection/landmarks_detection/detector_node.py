import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import torchvision.transforms as T
import cv2
import sys
from pathlib import Path
import os

DETR_ROOT = os.environ.get("DETR_ROOT")
if DETR_ROOT is None:
    raise RuntimeError(
        "DETR_ROOT environment variable is not set. "
        "Please export DETR_ROOT=/path/to/Deformable-DETR"
    )

DETR_ROOT = Path(DETR_ROOT)
sys.path.insert(0, str(DETR_ROOT))
from models import build_model
from util.misc import nested_tensor_from_tensor_list

# -----------------------

class DeformableDETRNode(Node):

    def __init__(self):
        super().__init__("deformable_detr_node")
        self.class_names = [
            "ball",
            "goal",
            "robot",
            "L",
            "T",
            "X",
            #"red_robot",
            #"blue_robot"
        ]

        self.declare_parameter("checkpoint")
        self.declare_parameter("image_topic", "/camera/front/image_raw")
        self.declare_parameter("visualize", True)
        self.visualize = self.get_parameter("visualize").value
        
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
            qos_profile_sensor_data
        )
        # self.transform = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406],
        #                 [0.229, 0.224, 0.225])
        # ])


    def load_model(self, checkpoint_path):
        self.get_logger().info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

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
        # img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)

        h, w, _ = img_rgb.shape
        orig_size = torch.tensor([[h, w]], device=self.device)

        results = self.postprocessors["bbox"](outputs, orig_size)[0]

        scores = results["scores"]
        labels = results["labels"]

        keep = scores > 0.6

        boxes  = results["boxes"][keep]
        labels = labels[keep]
        scores = scores[keep]


        for box, label, score in zip(boxes, labels, scores):
            x0, y0, x1, y1 = box.int().tolist()

            class_id = int(label.item())
            class_name = self.class_names[class_id]
            text = f"{class_name} {score:.2f}"

            cv2.rectangle(cv_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            cv2.putText(
                cv_img,
                text,
                (x0, max(y0 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


        #cv2.imshow("detections", cv_img)
        if self.visualize:
            cv2.imshow("detections", cv_img)
            cv2.waitKey(1)

        # cv2.waitKey(1)


def main():
    rclpy.init()
    node = DeformableDETRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
