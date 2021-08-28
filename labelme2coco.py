import json
import os
import sys
import glob
import argparse
import numpy as np
from time import gmtime, strftime
from shutil import copyfile, rmtree
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default="raw_data/labelme/images/",
                    help="path to your labelme images")
parser.add_argument("--ant_dir", type=str, default="raw_data/labelme/annotations/",
                    help="path to your labelme annotations")
parser.add_argument("--save_dir", type=str, default="coco_annot/coco/",
                    help="path to your coco dataset saving directory")
parser.add_argument("--val_percent", type=float, default="0",
                    help="proportion of validation data set in training data set")
parser.add_argument("--version", type=str,
                    help="version of data set.")
args = parser.parse_args()


def directory_checking():
    if not os.path.exists(os.path.join(args.save_dir, "images")):
        os.mkdir(os.path.join(args.save_dir, "images"))
    if not os.path.exists(os.path.join(args.save_dir, "annotations")):
        os.mkdir(os.path.join(args.save_dir, "annotations"))
    files = glob.glob(os.path.join(args.save_dir, "images/train2017/*"))
    for file in files:
        os.remove(file)
    files = glob.glob(os.path.join(args.save_dir, "images/val2017/*"))
    for file in files:
        os.remove(file)


def check_args():
    if not os.path.isdir(args.img_dir):
        parser.print_help()
        sys.stderr.write("args path error: No such directory: {}\n".format(args.img_dir))
        exit(1)
    if not os.path.isdir(args.ant_dir):
        parser.print_help()
        sys.stderr.write("args path error: No such directory: {}\n".format(args.ant_dir))
        exit(1)
    if not os.path.isdir(args.save_dir):
        parser.print_help()
        sys.stderr.write("args path error: No such directory: {}\n".format(args.save_dir))
        exit(1)
    if not args.version:
        parser.print_help()
        sys.stderr.write("args version error: you need specify a version num.")
        exit(1)


def coco_temp():
    with open("coco_annot/coco/coco_basic_info.json") as f:
        temp = json.load(f)

    temp["info"]["date_created"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return temp


def poly_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * (np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_bbox(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    left = x.min()
    right = x.max()
    top = y.min()
    bottom = y.max()
    return [left, top, right-left, bottom - top]


def coco_img_name(x):
    str_x = str(x)
    for _ in range(8 - len(str_x)):
        str_x = '0' + str_x
    return str_x
# ------------------------------------------------------------------


def id_name(_id, ext_name):
    name = str(_id)
    for _ in range(8 - len(name)):
        name = '0' + name
    name += "." + ext_name
    return name


def find_class(classes, aim):
    for cls in classes:
        if cls["name"] == aim:
            return cls["id"]
    raise KeyError("class_name \"{}\" not found. please check classes_name.json".format(aim))


def temp_add_categories(coco):
    with open("coco_annot/coco/classes_name.json") as f:
        category = json.load(f)
    for i, name in enumerate(category):
        category_coco = {
            "id": i + 1,
            "name": name,
            "supercategory": "None",
        }
        coco["categories"].append(category_coco)


def coco_add_img(coco, file_name):
    create_time = os.path.getmtime(file_name)
    create_time = strftime("%Y-%m-%d %H:%M:%S", gmtime(create_time))
    img = Image.open(file_name)

    image = {
        "id": len(coco["images"]) + 1,
        "width": img.width,
        "height": img.height,
        "file_name": os.path.basename(file_name),
        "license": 1,
        "flickr_url": file_name,
        "coco_url": file_name,
        "date_captured": create_time,
    }
    coco["images"].append(image)


def coco_add_ant(coco, ant):
    points = np.asarray(ant["points"])
    annotation = {
        "id": len(coco["annotations"]) + 1,
        "image_id": len(coco["images"]),
        "category_id": find_class(coco["categories"], ant["label"]),
        "segmentation": [points.flatten().tolist()],
        "area": poly_area(points),
        "bbox": get_bbox(points),
        "iscrowd": 0,
    }
    coco["annotations"].append(annotation)


def coco_add_instance(coco, img_path, ant_path):
    coco_add_img(coco, img_path)

    with open(ant_path) as f:
        ants_json = json.load(f)

    for ant in ants_json["shapes"]:
        coco_add_ant(coco, ant)


if __name__ == '__main__':
    check_args()
    directory_checking()

    img_files = glob.glob(os.path.join(args.img_dir, "*"))
    data_index = np.arange(len(img_files))
    np.random.shuffle(data_index)
    train_index = data_index[int(len(img_files) * args.val_percent):].tolist()
    valid_index = data_index[:int(len(img_files) * args.val_percent)].tolist()

    print("creating coco instance train...")
    instance_train = coco_temp()
    temp_add_categories(instance_train)
    for i in train_index:
        img_file = img_files[i]
        ant_file = os.path.join(args.ant_dir, os.path.basename(img_file).split('.')[0] + ".json")
        cpy_name = id_name(len(instance_train["images"]) + 1, img_file.split('.')[1])
        cpy_file = os.path.join(args.save_dir, "images/train2017/" + cpy_name)
        copyfile(img_file, cpy_file)
        coco_add_instance(instance_train, cpy_file, ant_file)

    print("creating coco instance valid...")
    instance_val = coco_temp()
    temp_add_categories(instance_val)
    for i in valid_index:
        img_file = img_files[i]
        ant_file = os.path.join(args.ant_dir, os.path.basename(img_file).split('.')[0] + ".json")
        cpy_name = id_name(len(instance_val["images"]) + 1, img_file.split('.')[1])
        cpy_file = os.path.join(args.save_dir, "images/val2017/" + cpy_name)
        copyfile(img_file, cpy_file)
        coco_add_instance(instance_val, cpy_file, ant_file)

    print("saving...")
    with open(os.path.join(args.save_dir, "annotations/instances_train2017.json"), 'w') as f:
        json.dump(instance_train, f, indent=4, sort_keys=True)
    with open(os.path.join(args.save_dir, "annotations/instances_val2017.json"), 'w') as f:
        json.dump(instance_val, f, indent=4, sort_keys=True)
