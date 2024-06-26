# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import shutil
from enum import Enum
from typing import Any, Dict, TypeVar
from logging import getLogger, basicConfig, INFO

import torch
from torch import nn

__all__ = [
    "accuracy", "make_directory", "save_checkpoint",
    "Summary", "AverageMeter", "ProgressMeter", 
    "load_state_dict_resnet"
]

V = TypeVar("V")

"""Logger for printing."""
_LOG = getLogger(__name__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() # transpose
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # _LOG.info(f'pred {pred.cpu().numpy()[0, 0]} gnd {target.cpu().numpy()[0]}')

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        

def load_state_dict_resnet(
        model: nn.Module,
        model_weights_path: str,
        start_epoch: int = None,
        best_acc1: float = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None
):
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    # Load model state dict. Extract the fitted model weights
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                    k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    # Overwrite the model weights to the current model
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    return model, start_epoch, best_acc1, optimizer, scheduler


def save_checkpoint(
        epoch: int,
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        is_best: bool = False,
        is_last: bool = False,
        freq: int = 5,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    if epoch % freq == 0 and epoch > 100:
        torch.save(state_dict, checkpoint_path)

    if is_best:
        torch.save(state_dict, os.path.join(results_dir, "best.pth.tar"))
        #shutil.copyfile(checkpoint_path, os.path.join(results_dir, "best.pth.tar"))
    if is_last:
        #shutil.copyfile(checkpoint_path, os.path.join(results_dir, "last.pth.tar"))
        torch.save(state_dict, os.path.join(results_dir, "last.pth.tar"))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print("\t".join(entries))
        return entries

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        # print(" ".join(entries))
        return entries

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
