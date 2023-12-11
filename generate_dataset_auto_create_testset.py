from os import listdir
from os.path import isfile, join
import random
import argparse
import os
import yaml
from datasets import Datasets


# Generate dataset with train.txt, test.txt, val.txt for listing all the path of training data
# Automatically generate new path on config.yaml

def set_dataset_path(new_path, config_yaml = 'config.yaml'):
    with open(config_yaml) as f:
        doc = yaml.safe_load(f)

    doc['data']['dataset_path'] = new_path

    with open(config_yaml, 'w') as f:
        yaml.dump(doc, f)


#############################################################
    """ 
    [Auto Generate Test Dataset]
    DATASET Folder
        |   
        |----category1 -- img1.jpg
        |              -- img2.jpg 
        |              -- etc      
        |   
        |----category2 -- img1.jpg
        |              -- etc
        |         
        |----etc
    """
#############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default="./game_data ", help='Annotated Dataset Path')
    parser.add_argument("--cfg", default="./config.yaml", help="path to config file")
    args = parser.parse_args()
    
    
    if not os.path.exists(args.cfg):
        raise Exception(f"{args.cfg} does not exist")
    else:
        cfg_path = args.cfg

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(args.path):
        raise Exception(f"{args.path} does not exist.")
    else:
        DSPATH = args.path
        

    dataset_name = cfg["data"]["type"]
    print(dataset_name)
    ds = Datasets.get(dataset_name)


    classes = list(ds.mapper.keys())
    dataset_path = {"train":[], "val":[], "test":[]}

    SAVE_PATH = f"{os.getcwd()}/dataset_txt"
    print("save path", SAVE_PATH)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)

    for c in classes:
        CATEGORY_PATH = os.path.join(DSPATH, c)
        if os.path.exists(CATEGORY_PATH):
            cur_class_imgs = listdir(CATEGORY_PATH)
            no_of_imgs = len(cur_class_imgs)
            cur_class_imgs = [os.path.join(CATEGORY_PATH,img) for img in cur_class_imgs]
            random.shuffle(cur_class_imgs)
            train_portion, val_portion, test_portion = cfg['data_proportion']['train'],cfg['data_proportion']['val'],cfg['data_proportion']['test']
            print(train_portion, val_portion, test_portion)
            assert train_portion + val_portion + test_portion == 1
            train_step = int(train_portion * no_of_imgs)
            val_step = int(val_portion/(val_portion+test_portion) * (no_of_imgs-train_step))
            dataset_path['train'].extend((cur_class_imgs[:train_step]))
            dataset_path['val'].extend(cur_class_imgs[train_step:train_step+val_step])
            dataset_path['test'].extend(cur_class_imgs[train_step+val_step:])

    print(cfg['data_proportion'])
    for i in cfg['data_proportion']:
        with open(f'{SAVE_PATH}/{i}.txt', 'w') as f:
            [f.write(tr+'\n') and print('\n', tr) for tr in dataset_path[i]]
        f.close()

    set_dataset_path(SAVE_PATH, config_yaml='config.yaml')
