import pandas as pd
import os
import PIL.Image as Image
import json


def convert_miap2coco(miap_csv, type, output_json):
    # Load the csv file
    df = pd.read_csv(miap_csv)
    data_type = type

    images = {}
    image_id = 1

    for ImageID in df["ImageID"].values:
        if ImageID not in images:
            try:
                width, height = Image.open(
                    os.path.join(
                        f"/home/prml/Dataset/MIAP/{data_type}", ImageID + ".jpg"
                    )
                ).size
                images[ImageID] = {
                    "file_name": ImageID + ".jpg",
                    "height": height,
                    "width": width,
                    "id": image_id,
                }
                image_id += 1
            except:
                print(f"Image {ImageID} not found.")

    annotations = []
    ann_id = 1
    for i, row in df.iterrows():
        ImageID = row["ImageID"]
        width, height = images[ImageID]["width"], images[ImageID]["height"]
        x, y, w, h = (
            int(row["XMin"] * width),
            int(row["YMin"] * height),
            int((row["XMax"] - row["XMin"]) * width),
            int((row["YMax"] - row["YMin"]) * height),
        )
        annotations.append(
            {
                "image_id": images[ImageID]["id"],
                "bbox": [x, y, w, h],
                "category_id": 1,
                "id": ann_id,
                "iscrowd": 0,
                "area": w * h,
            }
        )
        ann_id += 1

    print("Num of images:", len(images), "Num of objects:", len(annotations))

    # Create COCO format dictionary
    converted = {
        "info": {
            "description": "MIAP",
            "url": "https://storage.googleapis.com/openimages/web/extended.html#miap",
            "version": "1.0",
            "year": 2021,
            "contributor": "MIAP",
            "date_created": "2021-06-01",
        },
        "licenses": [],
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{"supercategory": "person", "id": 1, "name": "person"}],
    }

    # Convert COCO format dictionary to JSON
    coco_json = json.dumps(converted)

    # Save JSON to a file
    with open(output_json, "w") as f:
        f.write(coco_json)


if __name__ == "__main__":

    print("val:", len(os.listdir("/home/prml/Dataset/MIAP/val")))
    convert_miap2coco(
        miap_csv="/home/prml/Dataset/MIAP/annotations/open_images_extended_miap_boxes_val.csv",
        type="val",
        output_json="/home/prml/Dataset/MIAP/annotations/val.json",
    )

    print("train:", len(os.listdir("/home/prml/Dataset/MIAP/train")))
    convert_miap2coco(
        miap_csv="/home/prml/Dataset/MIAP/annotations/open_images_extended_miap_boxes_train.csv",
        type="train",
        output_json="/home/prml/Dataset/MIAP/annotations/train.json",
    )

    print("test:", len(os.listdir("/home/prml/Dataset/MIAP/test")))
    convert_miap2coco(
        miap_csv="/home/prml/Dataset/MIAP/annotations/open_images_extended_miap_boxes_test.csv",
        type="test",
        output_json="/home/prml/Dataset/MIAP/annotations/test.json",
    )
