import json
import os
import sys
import glob
import argparse
import numpy as np
from time import gmtime, strftime
from shutil import copyfile, rmtree

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


def coco_add_img(coco, img_id, img_path, width, height):
    create_time = os.path.getmtime(img_path)
    create_time = strftime("%Y-%m-%d %H:%M:%S", gmtime(create_time))
    file_name = os.path.basename(img_path)

    image = {
        "id": img_id,
        "width": width,
        "height": height,
        "file_name": file_name,
        "license": 1,
        "flickr_url": img_path,
        "coco_url": img_path,
        "date_captured": create_time,
    }
    coco["images"].append(image)


def coco_add_ant(coco, ant_id, img_id, shapes):
    for obj in shapes:
        ant_id += 1
        points = np.asarray(obj["points"])
        annotation = {
            "id": ant_id,
            "image_id": img_id,
            "category_id": find_class(coco["categories"], obj["label"]),
            "segmentation": points.flatten().tolist(),
            "area": poly_area(points),
            "bbox": get_bbox(points),
            "iscrowd": 1,
        }
        coco["annotations"].append(annotation)
    return ant_id


def make_instance(coco, data):
    ant_id = 0
    for i, paths in enumerate(data):
        with open(paths[1], 'r') as f:
            ant_json = json.load(f)

        coco_add_img(coco, i+1, cpy_file, ant_json["imageWidth"], ant_json["imageHeight"])
        ant_id = coco_add_ant(coco, ant_id, i+1, ant_json["shapes"])


def make_caption(coco, data):
    img_id = 0
    ant_id = 0
    for i, paths in enumerate(data):
        with open(paths[1], 'r') as f:
            ant_json = json.load(f)

        coco_add_img(coco, img_id, cpy_file, ant_json["imageWidth"], ant_json["imageHeight"])
        for obj in ant_json["shapes"]:
            ant_id += 1
            annotation = {
                "id": ant_id,
                "image_id": img_id,
                "caption": "labeled by {}. label name is {}".format(coco["info"]["contributor"], obj["label"]),
            }
            coco["annotations"].append(annotation)


def find_class(classes, aim):
    for cls in classes:
        if cls["name"] == aim:
            return cls["id"]
    return -1


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


if __name__ == '__main__':
    check_args()
    directory_checking()

    imgs_path = glob.glob(os.path.join(args.img_dir, "*"))
    data_index = np.arange(len(imgs_path))
    np.random.shuffle(data_index)
    train_index = data_index[int(len(imgs_path) * args.val_percent):].tolist()
    valid_index = data_index[:int(len(imgs_path) * args.val_percent)].tolist()

    train_data = list()
    for i, num in enumerate(train_index):
        img_basename = os.path.basename(imgs_path[num])
        ant_basename = img_basename.split('.')[0] + ".json"

        cpy_file = coco_img_name(i+1) + '.' + img_basename.split('.')[1]
        cpy_file = os.path.join(args.save_dir, "images/train2017/" + cpy_file)

        copyfile(imgs_path[num], cpy_file)

        train_data.append([cpy_file, os.path.join(args.ant_dir, ant_basename)])

    valid_data = list()
    for i, num in enumerate(valid_index):
        img_basename = os.path.basename(imgs_path[num])
        ant_basename = img_basename.split('.')[0] + ".json"

        cpy_file = coco_img_name(i + 1) + '.' + img_basename.split('.')[1]
        cpy_file = os.path.join(args.save_dir, "images/val2017/" + cpy_file)

        copyfile(imgs_path[num], cpy_file)

        valid_data.append([cpy_file, os.path.join(args.ant_dir, ant_basename)])

    print("creating coco instance train...")
    instance_train = coco_temp()
    temp_add_categories(instance_train)
    make_instance(instance_train, train_data)

    print("creating coco instance valid...")
    instance_val = coco_temp()
    temp_add_categories(instance_val)
    make_instance(instance_val, valid_data)

    print("creating coco caption train...")
    caption_train = coco_temp()
    make_caption(caption_train, train_data)

    print("creating coco caption valid...")
    caption_val = coco_temp()
    make_caption(caption_val, valid_data)

    print("saving...")
    with open(os.path.join(args.save_dir, "annotations/instance_train2017.json"), 'w') as f:
        json.dump(instance_train, f, indent=4, sort_keys=True)
    with open(os.path.join(args.save_dir, "annotations/instance_val2017.json"), 'w') as f:
        json.dump(instance_val, f, indent=4, sort_keys=True)
    with open(os.path.join(args.save_dir, "annotations/caption_train2017.json"), 'w') as f:
        json.dump(caption_train, f, indent=4, sort_keys=True)
    with open(os.path.join(args.save_dir, "annotations/caption_val2017.json"), 'w') as f:
        json.dump(caption_val, f, indent=4, sort_keys=True)
