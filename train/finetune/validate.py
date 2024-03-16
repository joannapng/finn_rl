import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,), stable=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if stable and (max(topk) > 1 or len(topk) > 1):
        raise RuntimeError("Stable implementation supports only topk = (1,)")

    with torch.no_grad():
        batch_size = target.size(0)
        if stable:
            pred = np.argmax(output.cpu().numpy(), axis=1)
            pred = torch.tensor(pred, device=target.device, dtype=target.dtype).unsqueeze(0)
        else:
            maxk = max(topk)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res


def validate(model, val_loader):
    top1 = AverageMeter('Acc@1', ':6.2f')

    def print_accuracy(top1, prefix=''):
        print('{}Avg acc@1 {top1.avg:2.3f}'.format(prefix, top1=top1))

    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            target = target.to(device)
            target = target.to(dtype)
            images = images.to(device)
            images = images.to(dtype)

            output = model(images)
            # measure accuracy
            acc1, = accuracy(output, target, stable=True)
            top1.update(acc1[0], images.size(0))

        print_accuracy(top1, 'Total:')
    return top1.avg.cpu().numpy()
    


