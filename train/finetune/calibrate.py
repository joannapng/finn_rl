import torch
from torch.utils.data import DataLoader
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode

def calibrate(args, model, calib_loader):
    # Setup calibration dataloaders
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with calibration_mode(model):
            for i, (images, target) in enumerate(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                model(images)

        '''
        with bias_correction_mode(model):
            for i, (images, target) in enumerate(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                model(images)
        '''
