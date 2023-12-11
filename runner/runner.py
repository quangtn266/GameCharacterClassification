import numpy as np
import torch
import os
import shutil
from torch.cuda.amp import GradScaler, autocast
from runner.metrics import accuracy
from optimizers import SAM

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch, init_lr, lr_step):
    lr =  init_lr * (0.1 ** (epoch // lr_step))       # //: return result from part can be divided. e.g. 11//5 = 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def save_checkpoint(state, is_best, filename, resume_path, version):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, f'model_best_{version}.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

def cal_weight_loss(output_loss, weight):
    return torch.div(torch.sum(torch.mul(output_loss, weight)), torch.sum(weight))

def train(train_loader, model, criterion, optimizer, epoch, half, logger, scaler=None):
    losses = AverageMeter()
    acc = AverageMeter()
    model.train()

    for i, (input, target, weight) in enumerate(train_loader):

        #input = input.float().cuda()
        #target = target.cuda()

        input = input.float()
        target = target

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        #weight = weight.cuda()#torch.rand(input.size(0)).cuda()
        weight = weight#torch.rand(input.size(0)).cuda()

        if scaler is not None:
            with autocast():
                output = model(input_var)
                loss = criterion(output, target_var)
        else:
            output = model(input_var)
            loss = criterion(output, target_var)

        prec = accuracy(output.data, target, topk=(1))
        
        losses.update(float(loss), input.size(0))
        acc.update(prec[0], input.size(0))
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:         
            if not isinstance(optimizer, SAM):
                loss.backward()
                optimizer.step()
            else:
                loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)
                #criterion(model(input_var), target_var).backward()
                loss2 = criterion(model(input_var), target_var)

                loss2.backward()
                optimizer.second_step(zero_grad=True)


        if i % 50 == 0:
            log_text = 'Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, top1=acc, optimizer=optimizer)   ###-----optimizer
            print(log_text)
            logger.write(log_text+'\n')
            logger.flush()

        input.cpu()
        target.cpu()
        torch.cuda.empty_cache()

    print("Total Train losses: %s" % losses.avg)  # ------------------
    print("Total Train accuracy: %s" % acc.avg)  # -------------------
    return loss, prec, losses, acc
                    
def validate(val_loader, model, criterion, logger):
    losses = AverageMeter()
    acc = AverageMeter()
    model.eval()

    for i, (input, target, weight) in enumerate(val_loader):
#        input = input.float().cuda()
#        target = target.cuda()

        input = input.float()
        target = target

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        #weight = weight.cuda()#torch.rand(input.size(0)).cuda()
        weight = weight

        #print('val weight: ', weight)

        with torch.no_grad():
            output = model(input_var)

        loss = criterion(output, target_var)
        #loss = cal_weight_loss(loss, weight)
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss, input.size(0))
        acc.update(prec1[0], input.size(0))
        if i % 10 == 0:
            log_text = 'Test: [{0}/{1}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), loss=losses,
                   top1=acc)
            print(log_text)
            logger.write(log_text+'\n')
            logger.flush()

        input.cpu()
        target.cpu()
        torch.cuda.empty_cache()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=acc))
    print("Validation losses: {loss.avg}".format(loss=losses))   #-------------------
    print("Validation accuracy: %s" % float(acc.avg))  #-------------------
    return acc.avg, losses, acc

def test(test_loader, model, labels): #, logger):
    losses = AverageMeter()
    acc = AverageMeter()
    result = {'predict':[], 'targets':[]}
    model.eval()

    from PIL import Image
    import shutil
    shutil.rmtree('check_wrong_match/wrong_match_images')
    os.makedirs('check_wrong_match/wrong_match_images', exist_ok=True)
    with open('check_wrong_match/wrong_match.log', 'w') as ff:
        for i, (input, target, weight) in enumerate(test_loader):
#            input = input.float().cuda()
#            target = target.cuda()

            input = input.float()
            target = target

            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            with torch.no_grad():
                output = model(input_var)
            #print(f'output is: {output}')
            #print(f'predict percent: {torch.sigmoid(output[0])}')
            # print(f'predict: {int(torch.argmax(output))}, actual: {int(target)}; '
            #       f'mud percentage: {torch.sigmoid(output[0])[0]*100:.2f}%, '
            #       f'mud percentage within 3 classes: '
            #       f'{torch.sigmoid(output[0])[0]/torch.sigmoid(output[0]).sum()*100:.2f}%\n\n')

            result['predict'].append(int(torch.argmax(output)))
            result['targets'].append(int(target))

            if int(torch.argmax(output)) != int(target):
                ff.write(f'For {i}, Predict: {labels[int(torch.argmax(output))]}, Actual: {labels[int(target)]}\n')
                img = input[0].cpu().permute(1, 2, 0).numpy()
                # import shutil
                # #shutil.rmtree('check_wrong_match/wrong_match_images')

                import cv2
                cv2.imwrite(f"check_wrong_match/wrong_match_images/{i}.jpg", img)
                
    return result['predict'], result['targets']

def predict(predict_loader, model):
    model.eval()

    res = {}
    for i, (path, input) in enumerate(predict_loader):

        input = input.float().cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        with torch.no_grad():
            output = model(input_var)
        res[path[0]] = int(torch.argmax(output))

    return res


class ConsoleLogger:
    def write(self, text):
        print(text, end='')
