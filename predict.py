import os
import argparse
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import timm
import torch

import yaml
from datasets import Datasets
from tools import obtain_default_model, obtain_ckpt_by_epoch

def preprocess(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (138, 138))
    img2tensor = torch.tensor(img, dtype=torch.float32)
    img2tensor = img2tensor.view(3,138,138)
    img2tensor = torch.unsqueeze(img2tensor, dim=0)
    return img2tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.yaml", help="path to config file")
    parser.add_argument("-l", "--latest", default=False, action='store_true' ,help='ignore path and directly obtain the latest one.')
    parser.add_argument("-e", "--epoch", default=0, type=int, help='select certain epoch result for test.')
    parser.add_argument("-m", "--model", default="./experiments/safetyjacketfromperson/2023-04-11/resnet18/2023-04-11_20-13-07_default/train/model_best_2023-04-11_20-13-07_default.pth.tar", help="path - trained model to test1")
    args = parser.parse_args()

    if not os.path.exists(args.cfg):
        raise Exception(f"{args.cfg} does not exist")
    cfg_path = args.cfg
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.latest:
        print('Latest Model Selected: ')
        args.model = obtain_default_model(cfg)

    if args.epoch:
        print(f'Epoch {args.epoch} Model Selected: ')
        args.model = obtain_ckpt_by_epoch(cfg, args.epoch)


    if not os.path.exists(args.model) or not len(args.model):
        raise Exception(f"{args.model} does not exist")

    print('model path:', args.model)

    checkpoint = torch.load(args.model)

    model_info = args.model.replace("./", "").split("/")
    dataset_name = model_info[1]
    ds = Datasets.get(dataset_name)
    no_of_classes = len(ds.labels)
    model = timm.create_model(model_name=model_info[3], checkpoint_path=args.model, pretrained=True, num_classes=no_of_classes)
    model.eval()

    import glob

    f = open("./dataset_txt/test.txt", "r")
    dir_test = "test_data"

    os.makedirs(dir_test, exist_ok=True)

    #import shutil
    #for x in f:
    #    x = x.replace("\n","")
    #    shutil.copy(x, dir_test)
        #img_dirs.append(x)

    labels = {0: "Draven", 1: "Evelynn", 2: "Ezreal", 3: "Fiora", 4: "Fizz", 5: "Garen", 6: "Galio",
              7: "Gragas", 8: "Graves", 9: "Janna", 10: "Jarvan_IV", 11: "Jax", 12: "Jhin", 13: "Jinx",
              14: "Katarina", 15: "Kennen", 16: "Leona", 17: "Lee_Sin", 18: "Lulu", 19: "Lux", 20: "Malphite",
              21: "Master_Yi", 22: "Miss_Fortune", 23: "Nami", 24: "Nasus", 25: "Olaf", 26: "Orianna", 27: "Pantheon",
              28: "Rakan", 29: "Rammus", 30: "Rengar", 31: "Seraphine", 32: "Shyvana", 33: "Singed", 34: "Sona",
              35: "Soraka", 36: "Teemo", 37: "Tristana", 38: "Tryndamere", 39: "Twisted_Fate", 40: "Varus", 41: "Vayne", 42: "Vi",
              43: "Xin_Zhao", 44: "Yasuo", 45: "Wukong", 46: "Zed", 47: "Ziggs", 48: "Dr._Mundo", 49: "Ahri", 50: "Akali",
              51: "Alistar", 52: "Amumu", 53: "Annie", 54: "Ashe", 55: "Aurelion_Sol", 56: "Blitzcrank", 57: "Braum",
              58: "Camille", 59: "Corki", 60: "Darius", 61: "Diana", 62: "KaiSa", 63: "KhaZix"
              }

    img_dirs = glob.glob("./test_data/*")
    with open('output.txt', 'w') as f:
        for i in img_dirs:
            names = i.split("/")
            imgname = names[len(names)-1]
            img2tensor = preprocess(i)
            output = model(img2tensor)
            label = int(torch.argmax(output))
            if label in labels:
                value_predict = labels[label]
                f.write(imgname+"    "+value_predict+"\n")
