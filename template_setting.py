import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument("--description", type=str,
                    help="introduce your data set simply")
parser.add_argument("--contributor", type=str,
                    help="your author name.")
parser.add_argument("--raw_path", type=str, default="raw_data/labelme/images",
                    help="raw image data path.")
args = parser.parse_args()

if __name__ == '__main__':
    with open("coco_annot/coco/coco_basic_info.json") as f:
        template = json.load(f)

    template["info"]["version"] = "0.0.1"

    if args.description:
        template["info"]["description"] = args.description
    if not template["info"]["description"]:
        sys.stderr.write("field empty error: field name \"template[\"info\"][\"description\"]\"\n")

    if args.contributor:
        template["info"]["contributor"] = args.contributor
        template["licenses"][0]["name"] = template["info"]["contributor"] + " Labelme License"
    if not template["info"]["contributor"]:
        sys.stderr.write("field empty error: field name \"template[\"info\"][\"contributor\"]\"\n")

    if not args.raw_path:
        template["licenses"][0]["url"] = args.raw_path

    with open("coco_annot/coco/coco_basic_info.json", 'w') as f:
        json.dump(template, f)
