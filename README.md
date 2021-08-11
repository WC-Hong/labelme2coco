# This is annotation translate from lebelme to coco format

### Getting start
1. Put your labelme images to ```./raw_data/labelme/images```
2. Put your labelme annotations to ```./raw_data/labelme/annotations```
note: annotation file name must same as image file name
3. Put your classes name into ```./coco_annot/coco/classes_name.json```
4. Run ```python templatesetting.py --description <discribe your dataset> --contributor <author name>``` to set coco dataset information
``` 
usage: template_setting.py [-h] [--version VERSION]
                           [--description DESCRIPTION]
                           [--contributor CONTRIBUTOR] [--raw_path RAW_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --description DESCRIPTION
                        introduce your data set simply
  --contributor CONTRIBUTOR
                        your author name.
  --raw_path RAW_PATH   raw image data path.
```
5. Run ```python labelme2coco.py --val_percent <float number between 0-1> --version <coco dataset version>```
```
usage: labelme2coco.py [-h] [--img_dir IMG_DIR] [--ant_dir ANT_DIR]
                       [--save_dir SAVE_DIR] [--val_percent VAL_PERCENT]
                       [--version VERSION]

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     path to your labelme images
  --ant_dir ANT_DIR     path to your labelme annotations
  --save_dir SAVE_DIR   path to your coco dataset saving directory
  --val_percent VAL_PERCENT
                        proportion of validation data set in training data set
  --version VERSION     version of data set.
```
6. Get your coco dataset in ```./coco_ant/coco```