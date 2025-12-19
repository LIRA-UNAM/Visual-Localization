#!/home/roboworks/Localization/bin/python
import rclpy
from rclpy.node import Node

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from torchvision import transforms
from argparse import Namespace

DETR_ROOT = Path.home() / "Visual-Localization" / "Deformable-DETR"
sys.path.insert(0, str(DETR_ROOT))
from models import build_model
from util.misc import nested_tensor_from_tensor_list


class DeformableDETRNode(Node):

    def __init__(self):
        super().__init__('detector_node')

        # ========== PARAMS ==========
        self.declare_parameter('checkpoint', 'checkpoint.pth')
        self.declare_parameter('conf_thresh', 0.6)
        self.declare_parameter('image_topic', '/camera/front/image_raw')

        self.checkpoint = self.get_parameter('checkpoint').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.image_topic = self.get_parameter('image_topic').value

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')

        # ========== MODEL ==========
        self.model = self.load_model()

        # ========== ROS ==========
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            10
        )

        self.pub_img = self.create_publisher(
            Image,
            '/deformable_detr/debug_image',
            10
        )

        self.get_logger().info('Deformable DETR node ready')

    # ===============================
    # BUILD MODEL (OFFICIAL REPO)
    # ===============================
    def get_args(self):
        return Namespace(
            backbone='resnet50',
            dilation=False,
            position_embedding='sine',
            num_feature_levels=4,
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=1024,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_queries=300,
            dec_n_points=4,
            enc_n_points=4,
            aux_loss=False,
            masks=False,
            frozen_weights=None,
            device=self.device,
        )

    def load_model(self):
        args = self.get_args()
        model, _, _ = build_model(args)

        ckpt = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(ckpt['model'], strict=False)

        model.to(self.device)
        model.eval()
        return model

    # ===============================
    # IMAGE CALLBACK
    # ===============================
    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tensor = transforms.ToTensor()(rgb)
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(tensor)

        samples = nested_tensor_from_tensor_list([tensor]).to(self.device)

        with torch.no_grad():
            outputs = self.model(samples)

        probs = outputs['pred_logits'].softmax(-1)[0]
        boxes = outputs['pred_boxes'][0]
        scores, labels = probs[..., :-1].max(-1)

        for score, label, box in zip(scores, labels, boxes):
            if score < self.conf_thresh:
                continue

            x1, y1, x2, y2 = self.cxcywh_to_xyxy(box, w, h)

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'cls:{label.item()} {score:.2f}',
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)

    # ===============================
    def cxcywh_to_xyxy(self, box, w, h):
        cx, cy, bw, bh = box.cpu().numpy()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        return x1, y1, x2, y2


def main():
    rclpy.init()
    node = DeformableDETRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
