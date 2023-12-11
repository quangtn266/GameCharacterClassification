import os
import argparse
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import shutil

from datasets import Datasets
from losses import Losses
from optimizers import Optimizers, SAM

from runner import AverageMeter, adjust_learning_rate, save_checkpoint, accuracy, train, validate, test
#from augmentations import get_aug_by_sizes2, augment_flips, strong_aug

import yaml

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from tools import obtain_default_model, obtain_ckpt_by_epoch

def sort_log(log='./check_wrong_match/wrong_match.log'):
    labels = []
    pred = []
    lines = []

    with open(log) as f:
        for line in f.readlines():
            term = line.split(', ')
            print(term)
            #id = term[0][4:]
            pred_label = term[1][9:]
            #actual_label = term[2][7:]
            if pred_label not in labels:
                labels.append(pred_label)
            pred.append(pred_label)
            lines.append(line)
    f.close()

    with open('./check_wrong_match/wrong_match_sort.log', 'w') as nf:
        for label in labels:
            for idx in range(len(pred)):
                if pred[idx] == label:
                    nf.write(lines[idx])
    nf.close()



def checkif_same_testset(path_exist, path_cur):
    # test.txt file exists - check if same -> then cover old result with the new one
    # test.txt file exists - diff - create new test1 dir, new result
    import filecmp
    dir_ls = os.listdir(path_exist)
    exist_test_dirs = [dir for dir in dir_ls if "test1" in dir]
    if not len(exist_test_dirs):
        return False, ''

    for test_dir in exist_test_dirs:
        path = f'{path_exist}/{test_dir}'
        if os.path.isfile(os.path.join(path,'test.txt')):
            if filecmp.cmp(os.path.join(path,'test.txt'), os.path.join(path_cur,'test.txt')):
                return True, path

    return False, exist_test_dirs[-1]

def plot_confusion_matrix(y_pred, y_actu):
    return pd.crosstab(pd.Series(y_actu), pd.Series(y_pred), rownames=['Actual'], colnames=['Predicted'], margins=True)

def perf_measure(y_actual, y_hat):
    T = 0
    F = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]:
           T += 1
        if y_hat[i]!=y_actual[i]:
           F += 1

    return T, F

########################################################################################################################################

#Please Use Labeled Dataset,
#Run
#python3 test.py -l                         -> Use the latest trained 'best-score-epoch' model to do inference on pre-generated testset
#python3 test.py -e 5                       -> Use the latest trained '5th epoch' model to do inference on pre-generated testset
#python3 test.py -m [relative-model-path]   -> Use specific model to do inference on pre-generated testset

# Check your test result on your corresponding testing model experiment folder 
# under ./experiments/[dataset_type]/[model_type]/[date+time]/test

#########################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.yaml", help="path to config file")
    parser.add_argument("-l", "--latest", default=False, action='store_true' ,help='ignore path and directly obtain the latest one.')
    parser.add_argument("-e", "--epoch", default=0, type=int, help='select certain epoch result for test.')
    parser.add_argument("-m", "--model", default="./experiments/yellowmudwater/2022-06-17/mixnet_s/2022-06-17_17-45-12_default/train/model_best_2022-06-17_17-45-12_default.pth.tar", help="path - trained model to test1")
    #parser.add_argument("--supervised", action='store_true', help='Doing folder classification task on non-labeled raw data')
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

    testset_exist, path = checkif_same_testset(f'{"/".join(model_info[:-2])}', cfg["data"]["dataset_path"])
    test_dir_num = ''
    if testset_exist:
        target_path = path
        print(f'Same test dataset exist, result on <{target_path}> was covered.')
    else:
        test_dir_num = str(int(path[4:])+1) if len(path) >0 else 1
        target_path = f'{"/".join(model_info[:-2])}/test{test_dir_num}'
        os.makedirs(target_path, exist_ok=True)
        print(f'Test with new dataset, new result on <{target_path}> was create.')
    logger = open(f'{target_path}/test.log', 'w')
    shutil.copy(os.path.join(cfg["data"]["dataset_path"], "test.txt"), f"./{target_path}/test.txt")

    size=cfg['input_size']    
    test_datasets = ds('test',
                       cfg["data"]["dataset_path"],
                       test_aug=False,
                       size=cfg["input_size"]
                       )

    ori_size = test_datasets.origin_size_group
    ori_size['predict_val'] = {}
    ori_size['actual_val'] = {}
    acc_rate = {}

    test_loader = DataLoader(test_datasets,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    model = timm.create_model(model_name=model_info[3], checkpoint_path=args.model, pretrained=True, num_classes=no_of_classes)
    #model.cuda().eval()
    model.eval()

    #print(vars(ds))
    labels = ds.labels
    y_pred, y_actu = test(test_loader, model, labels)
    confusion_matrix = plot_confusion_matrix(pd.Series(y_actu), pd.Series(y_pred))
    dfcm = confusion_matrix.to_string(header=True, index=True)
    T, F = perf_measure(y_actu, y_pred)

    logger.write(f"Confusion matrix by classes for all test1 data:\n {dfcm} \nTrue: {T}, False: {F}\nAccuracy: {float(T/(T+F)*100)}%\n\n")

    os.makedirs(os.path.join(target_path, 'confusion matrices(size)'), exist_ok=True)
    for size in ori_size['imgs'].keys():

        if not len(ori_size['imgs'][size]):
            continue
        ori_size['predict_val'][size] = []
        ori_size['actual_val'][size] = []
        indexes = ori_size['imgs'][size]
        [ori_size['predict_val'][size].append(y_pred[idx]) for idx in indexes]
        [ori_size['actual_val'][size].append(y_actu[idx]) for idx in indexes]
        T, F = perf_measure(ori_size['actual_val'][size], ori_size['predict_val'][size])
        ori_size['predict_val'][size] = pd.Series(ori_size['predict_val'][size])
        ori_size['actual_val'][size] = pd.Series(ori_size['actual_val'][size])

        # Confusion matrix, heatmap
        confusion_matrix = plot_confusion_matrix(ori_size['actual_val'][size], ori_size['predict_val'][size])
        dfcm = confusion_matrix.to_string(header=True, index=True)
        hm = sn.heatmap(confusion_matrix, annot=True, cbar=False, cmap="Blues", fmt='d')
        hm.figure.savefig(f'{target_path}/confusion matrices(size)/heatmap_size_>={size}.png')
        plt.clf()

        logger.write(f'For image with size larger than {size}, the confusion matrix is:\n')
        logger.write(f'{dfcm}\nTrue: {T}, False: {F}\nAccuracy: {float(T/(T+F)*100)}%\n\n')

        # Bar chart - Accuracy (image size)
        acc_rate[size] = int(T/ori_size['sz'][size]*100)

    #print(acc_rate)
    #print(list(acc_rate.values()))
    #print(list(acc_rate.keys()))
    
    logger.close()
    sort_log()
