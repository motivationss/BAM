from multiprocessing.util import info
import os, csv
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import datetime
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np
import wandb
from copy import deepcopy

import torch.nn.functional as F
from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from data import dro_dataset
from data import folds
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, hinge_loss, squared_loss, l1_loss, huber_loss
from train import train
from data.folds import Subset, ConcatDataset
import time
import subprocess
def generate_random(indices, nums_to_keep):
    return np.random.permutation(indices)[:nums_to_keep]

def subsample_val_data(args, val_data, subsample_proportion=1.0):
    if args.dataset == 'CelebA':
        metadata_path = "./celebA/data/metadata.csv"
    elif args.dataset == 'MultiNLI':
        metadata_path = "./multinli/data/metadata.csv"
    elif args.dataset == 'CUB':
        metadata_path = "./cub/data/waterbird_complete95_forest2water2/metadata.csv"
    elif args.dataset == "jigsaw":
        metadata_path = "./jigsaw/data/all_data_with_identities.csv"
    else: 
        raise NotImplementedError 

    original_df = pd.read_csv(metadata_path)
    dataset = args.dataset 
    if dataset == "jigsaw":
        original_val_df = original_df[original_df["split"] == "val"]
    else:
        original_val_df = original_df[original_df["split"] == 1]
    
    if dataset == "CUB":
        group_0_indices = np.where((original_val_df['y'] == 0) & (original_val_df['place'] == 0))[0]
        group_1_indices = np.where((original_val_df['y'] == 0) & (original_val_df['place'] == 1))[0]
        group_2_indices = np.where((original_val_df['y'] == 1) & (original_val_df['place'] == 0))[0]
        group_3_indices = np.where((original_val_df['y'] == 1) & (original_val_df['place'] == 1))[0]
        # print((len(group_1_indices) * 0.8 ))
        group_0_subsample = generate_random(group_0_indices, int(len(group_0_indices) * subsample_proportion))
        group_0_subset = Subset(val_data, list(group_0_subsample))
        group_1_subsample = generate_random(group_1_indices, int(len(group_1_indices) * subsample_proportion))
        group_1_subset = Subset(val_data, list(group_1_subsample))
        group_2_subsample = generate_random(group_2_indices, int(len(group_2_indices) * subsample_proportion))
        group_2_subset = Subset(val_data, list(group_2_subsample))
        group_3_subsample = generate_random(group_3_indices, int(len(group_3_indices) * subsample_proportion))
        group_3_subset = Subset(val_data, list(group_3_subsample))


        subsampled_val_data = dro_dataset.DRODataset(
            ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset]),
            process_item_fn=None,
            n_groups=val_data.n_groups,
            n_classes=val_data.n_classes,
            group_str_fn=val_data.group_str,
        )
    elif dataset == 'CelebA':
        group_0_indices = np.where((original_val_df['Blond_Hair'] == 0) & (original_val_df['Male'] == 0))[0]
        group_1_indices = np.where((original_val_df['Blond_Hair'] == 0) & (original_val_df['Male'] == 1))[0]
        group_2_indices = np.where((original_val_df['Blond_Hair'] == 1) & (original_val_df['Male'] == 0))[0]
        group_3_indices = np.where((original_val_df['Blond_Hair'] == 1) & (original_val_df['Male'] == 1))[0]
        # print((len(group_1_indices) * 0.8 ))
        group_0_subsample = generate_random(group_0_indices, int(len(group_0_indices) * subsample_proportion))
        group_0_subset = Subset(val_data, list(group_0_subsample))
        group_1_subsample = generate_random(group_1_indices, int(len(group_1_indices) * subsample_proportion))
        group_1_subset = Subset(val_data, list(group_1_subsample))
        group_2_subsample = generate_random(group_2_indices, int(len(group_2_indices) * subsample_proportion))
        group_2_subset = Subset(val_data, list(group_2_subsample))
        group_3_subsample = generate_random(group_3_indices, int(len(group_3_indices) * subsample_proportion))
        group_3_subset = Subset(val_data, list(group_3_subsample))


        subsampled_val_data = dro_dataset.DRODataset(
            ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset]),
            process_item_fn=None,
            n_groups=val_data.n_groups,
            n_classes=val_data.n_classes,
            group_str_fn=val_data.group_str,
        )
    elif dataset == 'jigsaw':
        group_0_indices = np.where((original_val_df['toxicity'] == 0) & (original_val_df['identity_any'] == 0))[0]
        group_1_indices = np.where((original_val_df['toxicity'] == 0) & (original_val_df['identity_any'] == 1))[0]
        group_2_indices = np.where((original_val_df['toxicity'] == 1) & (original_val_df['identity_any'] == 0))[0]
        group_3_indices = np.where((original_val_df['toxicity'] == 1) & (original_val_df['identity_any'] == 1))[0]
        # print((len(group_1_indices) * 0.8 ))
        group_0_subsample = generate_random(group_0_indices, int(len(group_0_indices) * subsample_proportion))
        group_0_subset = Subset(val_data, list(group_0_subsample))
        group_1_subsample = generate_random(group_1_indices, int(len(group_1_indices) * subsample_proportion))
        group_1_subset = Subset(val_data, list(group_1_subsample))
        group_2_subsample = generate_random(group_2_indices, int(len(group_2_indices) * subsample_proportion))
        group_2_subset = Subset(val_data, list(group_2_subsample))
        group_3_subsample = generate_random(group_3_indices, int(len(group_3_indices) * subsample_proportion))
        group_3_subset = Subset(val_data, list(group_3_subsample))


        subsampled_val_data = dro_dataset.DRODataset(
            ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset]),
            process_item_fn=None,
            n_groups=val_data.n_groups,
            n_classes=val_data.n_classes,
            group_str_fn=val_data.group_str,
        )
    elif dataset == 'MultiNLI':
        group_0_indices = np.where((original_val_df['gold_label'] == 0) & (original_val_df['sentence2_has_negation'] == 0))[0]
        group_1_indices = np.where((original_val_df['gold_label'] == 0) & (original_val_df['sentence2_has_negation'] == 1))[0]
        group_2_indices = np.where((original_val_df['gold_label'] == 1) & (original_val_df['sentence2_has_negation'] == 0))[0]
        group_3_indices = np.where((original_val_df['gold_label'] == 1) & (original_val_df['sentence2_has_negation'] == 1))[0]
        group_4_indices = np.where((original_val_df['gold_label'] == 2) & (original_val_df['sentence2_has_negation'] == 0))[0]
        group_5_indices = np.where((original_val_df['gold_label'] == 2) & (original_val_df['sentence2_has_negation'] == 1))[0]
        # print((len(group_1_indices) * 0.8 ))
        group_0_subsample = generate_random(group_0_indices, int(len(group_0_indices) * subsample_proportion))
        group_0_subset = Subset(val_data, list(group_0_subsample))
        group_1_subsample = generate_random(group_1_indices, int(len(group_1_indices) * subsample_proportion))
        group_1_subset = Subset(val_data, list(group_1_subsample))
        group_2_subsample = generate_random(group_2_indices, int(len(group_2_indices) * subsample_proportion))
        group_2_subset = Subset(val_data, list(group_2_subsample))
        group_3_subsample = generate_random(group_3_indices, int(len(group_3_indices) * subsample_proportion))
        group_3_subset = Subset(val_data, list(group_3_subsample))
        group_4_subsample = generate_random(group_4_indices, int(len(group_4_indices) * subsample_proportion))
        group_4_subset = Subset(val_data, list(group_4_subsample))
        group_5_subsample = generate_random(group_5_indices, int(len(group_5_indices) * subsample_proportion))
        group_5_subset = Subset(val_data, list(group_5_subsample))


        subsampled_val_data = dro_dataset.DRODataset(
            ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset, group_4_subset, group_5_subset]),
            process_item_fn=None,
            n_groups=val_data.n_groups,
            n_classes=val_data.n_classes,
            group_str_fn=val_data.group_str,
        )
    else:
        raise NotImplementedError

    return subsampled_val_data

def val_fixed_size(args, val_data, group_size = [50, 10, 10, 50]):
    if args.dataset == 'CelebA':
        metadata_path = "./celebA/data/metadata.csv"
    elif args.dataset == 'MultiNLI':
        metadata_path = "./multinli/data/metadata.csv"
    elif args.dataset == 'CUB':
        metadata_path = "./cub/data/waterbird_complete95_forest2water2/metadata.csv"
    elif args.dataset == "jigsaw":
        metadata_path = "./jigsaw/data/all_data_with_identities.csv"
    else: 
        raise NotImplementedError 

    original_df = pd.read_csv(metadata_path)
    dataset = args.dataset 
    if dataset == "jigsaw":
        original_val_df = original_df[original_df["split"] == "val"]
    else:
        original_val_df = original_df[original_df["split"] == 1]
    
    if dataset == "CUB":
        group_0_indices = np.where((original_val_df['y'] == 0) & (original_val_df['place'] == 0))[0]
        group_1_indices = np.where((original_val_df['y'] == 0) & (original_val_df['place'] == 1))[0]
        group_2_indices = np.where((original_val_df['y'] == 1) & (original_val_df['place'] == 0))[0]
        group_3_indices = np.where((original_val_df['y'] == 1) & (original_val_df['place'] == 1))[0]
        # print((len(group_1_indices) * 0.8 ))

        group_0_subsample_ = generate_random(group_0_indices, int(len(group_0_indices) * 0.1))
        print("after ratio: ", group_0_subsample_)
        group_0_subset_ = Subset(val_data, list(group_0_subsample_))
        print("success with subsample")
        
        print("group size 0: ",group_size[0])
        group_0_subsample = generate_random(group_0_indices, group_size[0])
        print("after fixed 0: ", group_0_subsample)
        group_0_subset = Subset(val_data, list(group_0_subsample))
        group_1_subsample = generate_random(group_1_indices, group_size[1])
        group_1_subset = Subset(val_data, list(group_1_subsample))
        group_2_subsample = generate_random(group_2_indices, group_size[2])
        group_2_subset = Subset(val_data, list(group_2_subsample))
        group_3_subsample = generate_random(group_3_indices, group_size[3])
        group_3_subset = Subset(val_data, list(group_3_subsample))


        subsampled_val_data = dro_dataset.DRODataset(
            ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset]),
            process_item_fn=None,
            n_groups=val_data.n_groups,
            n_classes=val_data.n_classes,
            group_str_fn=val_data.group_str,
        )
    # elif dataset == 'CelebA':
    #     group_0_indices = np.where((original_val_df['Blond_Hair'] == 0) & (original_val_df['Male'] == 0))[0]
    #     group_1_indices = np.where((original_val_df['Blond_Hair'] == 0) & (original_val_df['Male'] == 1))[0]
    #     group_2_indices = np.where((original_val_df['Blond_Hair'] == 1) & (original_val_df['Male'] == 0))[0]
    #     group_3_indices = np.where((original_val_df['Blond_Hair'] == 1) & (original_val_df['Male'] == 1))[0]
    #     # print((len(group_1_indices) * 0.8 ))
    #     group_0_subsample = generate_random(group_0_indices, int(len(group_0_indices) * subsample_proportion))
    #     group_0_subset = Subset(val_data, list(group_0_subsample))
    #     group_1_subsample = generate_random(group_1_indices, int(len(group_1_indices) * subsample_proportion))
    #     group_1_subset = Subset(val_data, list(group_1_subsample))
    #     group_2_subsample = generate_random(group_2_indices, int(len(group_2_indices) * subsample_proportion))
    #     group_2_subset = Subset(val_data, list(group_2_subsample))
    #     group_3_subsample = generate_random(group_3_indices, int(len(group_3_indices) * subsample_proportion))
    #     group_3_subset = Subset(val_data, list(group_3_subsample))


    #     subsampled_val_data = dro_dataset.DRODataset(
    #         ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset]),
    #         process_item_fn=None,
    #         n_groups=val_data.n_groups,
    #         n_classes=val_data.n_classes,
    #         group_str_fn=val_data.group_str,
    #     )
    # elif dataset == 'jigsaw':
    #     group_0_indices = np.where((original_val_df['toxicity'] == 0) & (original_val_df['identity_any'] == 0))[0]
    #     group_1_indices = np.where((original_val_df['toxicity'] == 0) & (original_val_df['identity_any'] == 1))[0]
    #     group_2_indices = np.where((original_val_df['toxicity'] == 1) & (original_val_df['identity_any'] == 0))[0]
    #     group_3_indices = np.where((original_val_df['toxicity'] == 1) & (original_val_df['identity_any'] == 1))[0]
    #     # print((len(group_1_indices) * 0.8 ))
    #     group_0_subsample = generate_random(group_0_indices, int(len(group_0_indices) * subsample_proportion))
    #     group_0_subset = Subset(val_data, list(group_0_subsample))
    #     group_1_subsample = generate_random(group_1_indices, int(len(group_1_indices) * subsample_proportion))
    #     group_1_subset = Subset(val_data, list(group_1_subsample))
    #     group_2_subsample = generate_random(group_2_indices, int(len(group_2_indices) * subsample_proportion))
    #     group_2_subset = Subset(val_data, list(group_2_subsample))
    #     group_3_subsample = generate_random(group_3_indices, int(len(group_3_indices) * subsample_proportion))
    #     group_3_subset = Subset(val_data, list(group_3_subsample))


    #     subsampled_val_data = dro_dataset.DRODataset(
    #         ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset]),
    #         process_item_fn=None,
    #         n_groups=val_data.n_groups,
    #         n_classes=val_data.n_classes,
    #         group_str_fn=val_data.group_str,
    #     )
    # elif dataset == 'MultiNLI':
    #     group_0_indices = np.where((original_val_df['gold_label'] == 0) & (original_val_df['sentence2_has_negation'] == 0))[0]
    #     group_1_indices = np.where((original_val_df['gold_label'] == 0) & (original_val_df['sentence2_has_negation'] == 1))[0]
    #     group_2_indices = np.where((original_val_df['gold_label'] == 1) & (original_val_df['sentence2_has_negation'] == 0))[0]
    #     group_3_indices = np.where((original_val_df['gold_label'] == 1) & (original_val_df['sentence2_has_negation'] == 1))[0]
    #     group_4_indices = np.where((original_val_df['gold_label'] == 2) & (original_val_df['sentence2_has_negation'] == 0))[0]
    #     group_5_indices = np.where((original_val_df['gold_label'] == 2) & (original_val_df['sentence2_has_negation'] == 1))[0]
    #     # print((len(group_1_indices) * 0.8 ))
    #     group_0_subsample = generate_random(group_0_indices, int(len(group_0_indices) * subsample_proportion))
    #     group_0_subset = Subset(val_data, list(group_0_subsample))
    #     group_1_subsample = generate_random(group_1_indices, int(len(group_1_indices) * subsample_proportion))
    #     group_1_subset = Subset(val_data, list(group_1_subsample))
    #     group_2_subsample = generate_random(group_2_indices, int(len(group_2_indices) * subsample_proportion))
    #     group_2_subset = Subset(val_data, list(group_2_subsample))
    #     group_3_subsample = generate_random(group_3_indices, int(len(group_3_indices) * subsample_proportion))
    #     group_3_subset = Subset(val_data, list(group_3_subsample))
    #     group_4_subsample = generate_random(group_4_indices, int(len(group_4_indices) * subsample_proportion))
    #     group_4_subset = Subset(val_data, list(group_4_subsample))
    #     group_5_subsample = generate_random(group_5_indices, int(len(group_5_indices) * subsample_proportion))
    #     group_5_subset = Subset(val_data, list(group_5_subsample))


        # subsampled_val_data = dro_dataset.DRODataset(
        #     ConcatDataset([group_0_subset, group_1_subset, group_2_subset, group_3_subset, group_4_subset, group_5_subset]),
        #     process_item_fn=None,
        #     n_groups=val_data.n_groups,
        #     n_classes=val_data.n_classes,
        #     group_str_fn=val_data.group_str,
        # )
    else:
        raise NotImplementedError

    return subsampled_val_data

def load_half_set(args, train_data, subsample_propo = 0.5):

    # subsample_propotion corresponds to how many samples you want to select

    metadata_df = pd.read_csv(args.metadata_path)
    
    indices = 2 # Indices of current train set
    subsample_indices = generate_random(indices, int(len(indices) * subsample_propo))
    subsample_points = Subset(train_data, list(subsample_indices))
    
    # see if we need to do this 
    # dro_dataset.DRODataset 

    return subsample_points

def subsample_upweight_confidence(args, train_data, upweight_fac, subsample_propo = 1.0, 
                            if_confidence=False, confidence_interval=0.5):
    """ Subsample non-error set, upweight error set with confidence to balance groups """
    assert args.aug_col is not None 
    metadata_df = pd.read_csv(args.metadata_path)
    if args.dataset == "jigsaw":
        train_col = metadata_df[metadata_df["split"] == "train"]
    else:
        train_col = metadata_df[metadata_df["split"] == 0]

    non_error_indices = np.where(train_col[args.aug_col] == 0)[0]
    subsample_indices = generate_random(non_error_indices, int(len(non_error_indices) * subsample_propo))
    subsample_points = Subset(train_data, list(subsample_indices))

    # error_part = train_col[train_col[args.aug_col] == 1]
    # error_part_pred_smaller_than_40 = error_part[error_part["confidence"] < 0.46]
    # error_part_pred_smaller_than_40 = np.where([error_part_pred_smaller_than_40["spurious"]])[0]
    # print("length of error sets that predict false greater than 0.4 that are on-spurious: ", len(error_part_pred_smaller_than_40))

    # print("original train_col: ", train_col)
    if if_confidence:
        # train_col = train_col[train_col["confidence"] < confidence_interval]
        error_indices = np.where(train_col["confidence"] < confidence_interval)[0]
        print("\n Note You are in Confidence Mode! :) \n")
    else:
        error_indices = np.where(train_col[args.aug_col] == 1)[0] # wrong_time
    # print("error indices: ", error_indices)
        ################ 2022.11.08
        # error_indices = np.where(train_col[args.error_set_col] == 1)[0]
    
    if args.minority_only:
        if args.dataset == "CUB":
            group_0_indices = np.where((train_col['y'] == 0) & (train_col['place'] == 0))[0]
            group_1_indices = np.where((train_col['y'] == 0) & (train_col['place'] == 1))[0]
            group_2_indices = np.where((train_col['y'] == 1) & (train_col['place'] == 0))[0]
            group_3_indices = np.where((train_col['y'] == 1) & (train_col['place'] == 1))[0]
            print("length of group 0: ", len(group_0_indices))
            print("length of group 1: ", len(group_1_indices))
            print("length of group 2: ", len(group_2_indices))
            print("length of group 3: ", len(group_3_indices)) 

            # print(type(group_0_indices))
            group_0_indices = list(group_0_indices)
            group_3_indices = list(group_3_indices)
            group_1_indices = list(group_1_indices)
            group_2_indices = list(group_2_indices)
            # print("can turn into list, ", len(group_0_indices))
            error_indices = group_1_indices + group_2_indices
            non_error_indices = group_0_indices + group_3_indices
            print("non error set - minority only - length: ",len(non_error_indices))
            print("error set - majority only - length: ", len(error_indices))
            subsample_indices = generate_random(non_error_indices, int(len(non_error_indices) * subsample_propo))
            subsample_points = Subset(train_data, list(subsample_indices))

    error_points = Subset(train_data, list(error_indices) * upweight_fac)

    subsample_data = dro_dataset.DRODataset(
        ConcatDataset([error_points, subsample_points]),
        process_item_fn=None,
        n_groups=train_data.n_groups,
        n_classes=train_data.n_classes,
        group_str_fn=train_data.group_str,
    )
    print("\n"*2)
    print("length of error set: ", len(error_indices))
    print("length of subsampled_non_error_set", len(subsample_indices))
    print("length of train data: ", len(subsample_data))
    print("downsubample coefficient: ", subsample_propo)
    print("upweight coefficient: ", upweight_fac)
    print("\n"*2)
    return subsample_data

def main(args):
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    if args.method == "AUX1":
        print("##################### Stage One Bis Amplification ##########################")
    elif args.method == "ProcessTrain":
        print("##################### Process Train ################################")
    elif args.method == "AUX2":
        print("###################### Stage Two Robust Training ####################")
        print(f"###################### UpWeight {args.up_weight} ###################")
        print(f"###################### Load Model {args.loadModel} #################")
    
    print("args: ", args)

    args.wandb = False

    if args.wandb:
        wandb.init(project=f"{args.project_name}_{args.dataset}")
        wandb.config.update(args)
    else:
        print("We are not using wandb!")

    if args.method == "ProcessTrain":
        info_dict = {}
        with open(os.path.join(args.log_dir_old, 'model_outputs/information.txt'), 'r') as f:
            for line in f:
                (k, v) = line.split()
                info_dict[k] = v
        args.lr = float(info_dict["lr"])
        args.n_epochs = int(info_dict["n_epochs"])
        args.weight_decay = 0.0
        previous_n_epochs = args.n_epochs
        
    # BERT-specific configs copied over from run_glue.py
    if (args.model.startswith("bert") and args.use_bert_params): 
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    mode = "w"

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, "log.txt"), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == "confounder":
        train_data, val_data, test_data = prepare_data(
            args,
            train=True,
        )

    elif args.shift_type == "label_shift_step":
        raise NotImplementedError
        train_data, val_data = prepare_data(args, train=True)
    
    print("length of train_data: ", len(train_data))
    print("length of test_data: ", len(test_data))
    print("length of val_data: ", len(val_data))
    
    print("args fold: ", args.fold)
    # print("args.up_weight: ", args.upweight)
    #########################################################################
    ###################### Prepare data for our method ######################
    #########################################################################

    # Should probably not be upweighting if folds are specified.
    # assert not args.fold or not args.up_weight
    
    if args.method == 'AUX2':
        train_data = subsample_upweight_confidence(args, train_data, upweight_fac = args.up_weight, 
        subsample_propo = args.subsample_propotion, if_confidence=args.use_confidence, 
        confidence_interval=args.confidence)
        

    #########################################################################
    #########################################################################
    #########################################################################
    
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    train_loader = dro_dataset.get_loader(train_data,
                                          train=True,
                                          reweight_groups=args.reweight_groups,
                                          **loader_kwargs)

    val_loader = dro_dataset.get_loader(val_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)

    if test_data is not None:
        test_loader = dro_dataset.get_loader(test_data,
                                             train=False,
                                             reweight_groups=None,
                                             **loader_kwargs)

    data = {}
    data["train_loader"] = train_loader
    data["val_loader"] = val_loader
    data["test_loader"] = test_loader
    data["train_data"] = train_data
    data["val_data"] = val_data
    data["test_data"] = test_data

    n_classes = train_data.n_classes

    log_data(data, logger)


    logger.flush()

    from utils import GeneralizedCELoss

    # GCE, CrossEntropy, Squared Loss, LabelSmoothing
    if args.loss == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
    elif args.loss == "SquaredLoss" or args.loss == "LabelSmoothingSquaredLoss":
        criterion = squared_loss
    elif args.loss == "LabelSmoothing":
        criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
    elif args.loss == "GCE":
        criterion = GeneralizedCELoss()
    elif args.loss == "L1Loss":
        criterion = l1_loss
    elif args.loss == "HuberLoss":
        # criterion = torch.nn.HuberLoss(reduction='none')
        criterion = huber_loss
    else:
        raise NotImplementedError
    
    
    epoch_offset = args.best_epoch if args.process_training else 0 
    if args.process_training:
        args.n_epochs = 0 
    
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"train.csv"),
                                      train_data.n_groups,
                                      mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"val.csv"),
                                    val_data.n_groups,
                                    mode=mode)
    test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv"),
                                     test_data.n_groups,
                                     mode=mode)

    if args.method == "ProcessTrain" and args.ProcessWhole:
        for idx in range(int(previous_n_epochs)):
            print(f"\n\n We are in Epoch {idx}\n\n")
            epoch_offset = idx 
            args.loadModel = str(idx)
            model = get_model(
                model=args.model,
                pretrained=not args.train_from_scratch,
                resume=args.resume,
                n_classes=train_data.n_classes,
                dataset=args.dataset,
                log_dir=args.log_dir_old,
                args=args,
                use_weighted_spurious_score = args.use_weighted_spurious_score,
                loader_len=len(data['train_data'])+len(data['val_data'])+len(data['test_data']) if args.method == 'AUX2' else None
            )
            train(
                model,
                criterion,
                data,
                logger,
                train_csv_logger,
                val_csv_logger,
                test_csv_logger,
                args,
                epoch_offset=epoch_offset,
                csv_name=args.fold,
                wandb=wandb if args.wandb else None,
            )
    else:
        model = get_model(
            model=args.model,
            pretrained=not args.train_from_scratch,
            resume=args.resume,
            n_classes=train_data.n_classes,
            dataset=args.dataset,
            log_dir=args.log_dir_old,
            args=args,
            use_weighted_spurious_score = args.use_weighted_spurious_score,
            loader_len=len(data['train_data'])+len(data['val_data'])+len(data['test_data']) if args.method == 'AUX2' else None
        )
        train(
            model,
            criterion,
            data,
            logger,
            train_csv_logger,
            val_csv_logger,
            test_csv_logger,
            args,
            epoch_offset=epoch_offset,
            csv_name=args.fold,
            wandb=wandb if args.wandb else None,
        )

    # test if we can use final result
    if args.method == "AUX1":
        torch.save(model, os.path.join(args.log_dir,"last_model.pth"))

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

def check_args(args):
    if args.shift_type == "confounder":
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith("label_shift"):
        assert args.minority_fraction
        assert args.imbalance_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument("-d",
                        "--dataset",
                        choices=dataset_attributes.keys(),
                        required=True)
    parser.add_argument("-s",
                        "--shift_type",
                        choices=shift_types,
                        required=True)
                        
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="spurious", help="wandb project name")
    # Confounders
    parser.add_argument("-t", "--target_name")
    parser.add_argument("-c", "--confounder_names", nargs="+")
    parser.add_argument("--up_weight", type=int, default=0)
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")
    # Label shifts
    parser.add_argument("--minority_fraction", type=float)
    parser.add_argument("--imbalance_ratio", type=float)
    # Data
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--reweight_groups", action="store_true",
                        default=False,
                        help="set to True if loss_type is group DRO")
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Objective
    parser.add_argument("--loss_type", default="erm",
                        choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--generalization_adjustment", default="0.0")
    parser.add_argument("--automatic_adjustment",
                        default=False,
                        action="store_true")
    parser.add_argument("--robust_step_size", default=0.01, type=float)
    parser.add_argument("--joint_dro_alpha", default=1, type=float,
                         help=("Size param for CVaR joint DRO."
                               " Only used if loss_type is joint_dro"))
    parser.add_argument("--use_normalized_loss",
                        default=False,
                        action="store_true")
    parser.add_argument("--btl", default=False, action="store_true")
    parser.add_argument("--hinge", default=False, action="store_true")
    # Model
    parser.add_argument("--model",
                        choices=model_attributes.keys(),
                        default="resnet50")
    parser.add_argument("--train_from_scratch",
                        action="store_true",
                        default=False)
    parser.add_argument('--aux_lambda', type=float, default=None)
    # Method
    parser.add_argument("--method", type=str, default="JTT")
    # Optimization
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--minimum_variational_weight", type=float, default=0)
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--log_dir_old", type=str, default=None)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument("--use_bert_params", type=int, default=1)
    parser.add_argument("--num_folds_per_sweep", type=int, default=5)
    parser.add_argument("--num_sweeps", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.7)

    parser.add_argument(
        "--metadata_csv_name",
        type=str,
        default="metadata.csv",
        help="name of the csv data file (dataset csv has to be placed in dataset folder).",
    )
    parser.add_argument("--fold", default=None)
    # Our groups (upweighting/dro_ours)
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="path to metadata csv",
    )
    parser.add_argument("--aug_col", default=None)

    # AUX2 input
    parser.add_argument("--use_weighted_spurious_score", default=False, action="store_true")
    parser.add_argument("--use_confidence", default=False, action="store_true")
    parser.add_argument("--confidence", default=0.5, type=float)
    parser.add_argument("--subsample_propotion", default=1.0, type=float)
    # parser.add_argument("--resume", default=False, action="store_true")

    # Process training put 
    parser.add_argument("--process_training", default=False, action="store_true")
    parser.add_argument("--best_epoch", type=int, default=0)
    parser.add_argument("--loadModel", type=str, default=None)
    parser.add_argument("--ProcessWhole", default=False, action="store_true")
    parser.add_argument("--load_new_model", action="store_true", default=False)
    # Loss type
    parser.add_argument("--loss", type=str, default="SquaredLoss", 
    help="GCE, CrossEntropy, SquaredLoss, LabelSmoothing")
    parser.add_argument("--minority_only", default=False, action="store_true")
    parser.add_argument("--val_subsample", type=float, default=1.0)

    # Control group
    parser.add_argument("--val_group0", type=int, default=467)
    parser.add_argument("--val_group1", type=int, default=466)
    parser.add_argument("--val_group2", type=int, default=133)
    parser.add_argument("--val_group3", type=int, default=133)
    parser.add_argument("--control_val_group", default=False, action="store_true")

    ###################### 2022.11.08

    parser.add_argument('--error_set_col', type=str, default='percentage')
    #percentage_list = [0.920, 0.930, 0.940, 0.950,
                    #    0.952, 0.954, 0.956, 0.958, 0.960, 0.970, 0.980]
                    
    # --error_set_col percentage_0.956
    ######################
    args = parser.parse_args()
    # print("cur")
    
    if args.model.startswith("bert"): # and args.model != "bert": 
        if args.use_bert_params:
            print("\n"*5, f"Using bert params", "\n"*5)
        else: 
            print("\n"*5, f"WARNING, Using {args.model} without using BERT HYPER-PARAMS", "\n"*5)

    check_args(args)
    if args.metadata_csv_name != "metadata.csv":
        print("\n" * 2
              + f"WARNING: You are using '{args.metadata_csv_name}' instead of the default 'metadata.csv'."
              + "\n" * 2)
    
    main(args)