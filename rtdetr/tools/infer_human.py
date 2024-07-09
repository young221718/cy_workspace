import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS

from PIL import Image

import argparse
import random
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T

import numpy as np
import torch
import src.misc.dist as dist

torch.set_grad_enabled(False)


def convert_cwh_to_xyxy(box):
    cx, cy, w, h = box[:, :, 0], box[:, :, 1], box[:, :, 2], box[:, :, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=2)


# COCO classes
CLASSES = ["person"]

# img_folder_path = "/home/prml/Dataset/MIAP/test"
# img_path_list = file_list = os.listdir(img_folder_path)[:10]

img_folder_path = "/home/prml/Dataset/CrowdHuman/val/"
img_path_list = [
    "273275,5d0330005d00a4c8.jpg",
    "273275,6bf8b00099f5b5c0.jpg",
    "273275,11c0bd000afcb7dbd.jpg",
    "273275,76cb60005cb0393a.jpg",
]

# standard PyTorch mean-std input image normalization
transform = T.Compose(
    [
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def main(args):
    dist.init_distributed()
    device = "cuda"

    cfg = YAMLConfig(
        args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.setup()
    solver.eval()

    model = solver.model
    model.to(device)

    model.eval()

    # ### ===================================================
    # ### end model setting
    # ### ==================================================

    for img_name in img_path_list:
        img_path = os.path.join(img_folder_path, img_name)
        im = Image.open(img_path)
        # im = im.crop((0, 0, 640, 640))
        img = transform(im).unsqueeze(0)
        img = img.cuda()

        # # propagate through the model
        outputs = model(img)
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 300, dim=1
        )
        scores = topk_values

        labels = topk_indexes % out_logits.shape[2]
        boxes = convert_cwh_to_xyxy(out_bbox)

        source_img = Image.open(img_path).convert("RGBA")
        img_w, img_h = source_img.size

        score_threshod = 0.5

        boxes = boxes[0]
        target_sizes = torch.tensor([[img_h, img_w]])
        target_sizes = target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        labels = labels[0]
        scores = scores[0]

        # plot_results
        source_img = Image.open(img_path).convert("RGBA")
        labels = labels.tolist()
        boxes = boxes[0].tolist()

        draw = ImageDraw.Draw(source_img)
        for (xmin, ymin, xmax, ymax), label, score in zip(boxes, labels, scores):
            if score > score_threshod:
                box_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=box_color, width=3)
                draw.text((xmin, ymin), text=str(CLASSES[label]))
            elif (xmax - xmin) * (ymax - ymin) < 40000:
                draw.rectangle(
                    ((xmin, ymin), (xmax, ymax)), outline=(0, 255, 0), width=1
                )
            else:
                draw.rectangle(
                    ((xmin, ymin), (xmax, ymax)), outline=(255, 0, 0), width=1
                )
                continue

        source_img.save(
            f"/home/prml/StudentsWork/ChanYoung/cy_workspace/detr/img_save/crowd_human/{img_name}_rtdetr_result2.png",
            "png",
        )

    print("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--tuning",
        "-t",
        type=str,
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    main(args)
