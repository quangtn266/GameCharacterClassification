import os
import glob

def obtain_default_model(cfg):
    dates = os.listdir(f'./experiments/{cfg["data"]["type"]}')
    dates.sort(reverse=True)
    folders = os.listdir(f'./experiments/{cfg["data"]["type"]}/{dates[0]}/{cfg["model"]}')
    folders.sort(reverse=True)

    model_path = ''
    for model in glob.glob(f'./experiments/{cfg["data"]["type"]}/{dates[0]}/'
                           f'{cfg["model"]}/{folders[0]}/train/*.pth.tar'):
        if 'model_best' in model:
            model_path = model

    return model_path

def obtain_ckpt_by_epoch(cfg, epoch):

    dates = os.listdir(f'./experiments/{cfg["data"]["type"]}')
    dates.sort(reverse=True)
    folders = os.listdir(f'./experiments/{cfg["data"]["type"]}/{dates[0]}/{cfg["model"]}')
    folders.sort(reverse=True)

    model_path = ''
    import glob
    for model in glob.glob(f'./experiments/{cfg["data"]["type"]}/{dates[0]}/'
                           f'{cfg["model"]}/{folders[0]}/train/*.pth.tar'):
        if str(epoch).zfill(3) in model:
            model_path = model

    return model_path