import sys
import os
import torch
import numpy as np
import csv

import torch
import torch.nn as nn
import torchvision
from models import model_attributes
import torch.nn.functional as F 
# console = sys.stdout

class Logger(object):
    def __init__(self, fpath=None, mode="w"):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
class Loggers(object):
    def __init__(self, fpath=None, mode="w"):
        
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        # console.write(msg)
        print(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        # console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # console.close()
        if self.file is not None:
            self.file.close()


class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode="w"):
        columns = ["epoch", "batch"]
        for idx in range(n_groups):
            columns.append(f"avg_loss_group:{idx}")
            columns.append(f"exp_avg_loss_group:{idx}")
            columns.append(f"avg_acc_group:{idx}")
            columns.append(f"processed_data_count_group:{idx}")
            columns.append(f"update_data_count_group:{idx}")
            columns.append(f"update_batch_count_group:{idx}")
        columns.append("avg_actual_loss")
        columns.append("avg_per_sample_loss")
        columns.append("avg_acc")
        columns.append("model_norm_sq")
        columns.append("reg_loss")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == "w":
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict["epoch"] = epoch
        stats_dict["batch"] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write("\n")

import torch.nn.functional as F
class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.6):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss


def huber_loss(logit, y, is_Multi=False):
    """ Implement L1 Loss"""
    
    num_class = 2 if not is_Multi else 3 
    expected_y = F.one_hot(y, num_classes=num_class)
    loss = torch.nn.HuberLoss(reduction='none')

    return loss(logit, expected_y.float()).sum(dim=1)

def l1_loss(logit, y, is_Multi=False):
    """ Implement L1 Loss"""

    num_class = 2 if not is_Multi else 3 
    expected_y = F.one_hot(y, num_classes=num_class)
    loss = torch.nn.L1Loss(reduction='none')
    return loss(logit, expected_y.float()).sum(dim=1)

def squared_loss(logit, y, is_Multi=False):
    """ Implement Squared Loss"""

    num_class = 2 if not is_Multi else 3 
    expected_y = F.one_hot(y, num_classes=num_class)
    loss = torch.nn.MSELoss(reduction='none')
    # loss = torch.square(logit - expected_y).sum(dim=1)
    return loss(logit, expected_y.float()).sum(dim=1)

    # exit()
    # if dataset_name == "MultiNLI":
    #     criteria = y.unsqueeze(-1)
    #     first_axe = torch.where(criteria == 0, 1, 0)
    #     second_axe = torch.where(criteria == 1, 1, 0)
    #     third_axe = torch.where(criteria==2, 1, 0)
    #     expected_y = torch.hstack((first_axe, second_axe, third_axe))
    #     loss = torch.square(logit - expected_y).sum(dim=1)
    # else:
    #     second_axe = y.unsqueeze(-1)
    #     first_axe = torch.where(second_axe > 0, 0, 1)
    #     expected_y = torch.hstack((first_axe, second_axe))
    #     loss = torch.square(logit - expected_y).sum(dim=1)
    # return loss

def hinge_loss(yhat, y):
    # The torch loss takes in three arguments so we need to split yhat
    # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
    # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
    # so we need to swap yhat[:, 0] and yhat[:, 1]...
    torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)


def get_model(model, pretrained, resume, n_classes, dataset, log_dir, args, 
              use_weighted_spurious_score=False, loader_len=None):
    # if resume:
        # model = torch.load(os.path.join(log_dir, "last_model.pth"))
        # d = train_data.input_size()[0]
    if model_attributes[model]["feature_type"] in (
            "precomputed",
            "raw_flattened",
    ):
        assert pretrained
        # Load precomputed features
        # d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model.startswith('bert'):
        if dataset == "MultiNLI":
            
            assert dataset == "MultiNLI"

            from pytorch_transformers import BertConfig, BertForSequenceClassification

            config_class = BertConfig
            model_class = BertForSequenceClassification

            config = config_class.from_pretrained("bert-base-uncased",
                                                num_labels=3,
                                                finetuning_task="mnli")
            model = model_class.from_pretrained("bert-base-uncased",
                                                from_tf=False,
                                                config=config)
        elif dataset == "jigsaw":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else: 
            raise NotImplementedError
    else:
        raise ValueError(f"{model} Model not recognized.")
    
    if not args.load_new_model:
        # if (args.method == 'AUX2' or args.method == 'ProcessTrain') and not use_weighted_spurious_score:
        #     print("YOU ARE USING BEST UNWEIGHTED MODEL!")
        #     model = torch.load(os.path.join(log_dir, "model_outputs/AUX1_best_model.pth"))
        #     # model = torch.load(os.path.join(log_dir, "model_outputs/400_model.pth"))
        #     # model = torch.load(os.path.join(log_dir, "model_outputs/450_model.pth"))
        # elif (args.method == 'AUX2' or args.method == 'ProcessTrain') and use_weighted_spurious_score:
        #     print("YOU ARE USING BEST WEIGHTED MODEL!")
        #     model = torch.load(os.path.join(log_dir, "model_outputs/AUX1_best_weighted_test_model.pth"))
        
        if args.loadModel is not None:
            print(f"We are using {args.loadModel}'s Model!")
            model = torch.load(os.path.join(log_dir, f"model_outputs/{args.loadModel}_model.pth"))
            if os.path.exists(os.path.join(log_dir, f"model_outputs/labelSmoothing_SqauredLoss_{args.loadModel}_model.pth")) and (args.loss != "LabelSmoothingSquaredLoss"):
                print(f"We are using labelSmoothing_SqauredLoss_{args.loadModel}_model.pth!")
                model = torch.load(os.path.join(log_dir, f"model_outputs/labelSmoothing_SqauredLoss_{args.loadModel}_model.pth"))
    else:
        print("You are using a new Model!")
    
    return model