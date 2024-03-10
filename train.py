import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

device = torch.device("cuda")
import pandas as pd
import os
import time 
import datetime
from datetime import datetime


def run_epoch(
    epoch,
    model,
    optimizer,
    loader,
    loss_computer,
    logger,
    csv_logger,
    args,
    is_training,
    process_training = False,
    show_progress=False,
    log_every=50,
    scheduler=None,
    csv_name=None,
    wandb_group=None,
    wandb=None,
):
    """
    scheduler is only used inside this function if model is bert.
    """

    # lamd = 100
    if is_training and (args.method != 'ProcessTrain' or args.loss == "LabelSmoothingSquaredLoss"):
        model.train()
        if (args.model.startswith("bert") and args.use_bert_params): # or (args.model == "bert"):
            model.zero_grad()
        
    else:
        print("model is evaluating!")
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training and (args.method != 'ProcessTrain' or args.loss == "LabelSmoothingSquaredLoss")):

        for batch_idx, batch in enumerate(prog_bar_loader):
            
            # db = model.b.grad
            # print('\n')
            # if db is not None:
            #     print("db sum: ", db.sum())
            # print('\n')
            
            # curr_b_sum = model.b.sum()
            # print("\n"*3, "current b sum: ", curr_b_sum, "\n"*3)
            
    
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            data_idx = batch[3]

            # print("x shape: ", x.shape)
            # print("y shape: ", y.shape)
            # print("g type: ", type(g))
            # print("data_idx: ", data_idx)
            
            if args.model.startswith("bert"):
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[1]  # [1] returns logits
                
            else:
                # outputs.shape: (batch_size, num_classes)
                outputs = model(x)
            
            if is_training and args.method == 'AUX1' and not process_training:
                for idx, i in enumerate(data_idx.detach().tolist()):
                    outputs[idx] += model.b[i] * args.aux_lambda
            # b[0]
                
            output_df = pd.DataFrame()

            # Calculate stats
            if batch_idx == 0:
                acc_y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc_y_true = y.cpu().numpy()
                indices = data_idx.cpu().numpy()
                
                probs = outputs.detach().cpu().numpy()
            else:
                acc_y_pred = np.concatenate([
                    acc_y_pred,
                    np.argmax(outputs.detach().cpu().numpy(), axis=1)
                ])
                acc_y_true = np.concatenate([acc_y_true, y.cpu().numpy()])
                indices = np.concatenate([indices, data_idx.cpu().numpy()])
                probs = np.concatenate([probs, outputs.detach().cpu().numpy()], axis = 0)
                
            assert probs.shape[0] == indices.shape[0]
            # TODO: make this cleaner.
            run_name = f"{csv_name}_epoch_{epoch}_val"
            output_df[f"y_pred_{run_name}"] = acc_y_pred
            output_df[f"y_true_{run_name}"] = acc_y_true
            output_df[f"indices_{run_name}"] = indices
            
            for class_ind in range(probs.shape[1]):
                output_df[f"pred_prob_{run_name}_{class_ind}"] = probs[:, class_ind]

            loss_main = loss_computer.loss(outputs, y, g, is_training)
            
            # if args.method == 'ProcessTrain':
            #     model.zero_grad()
            #     print("You are process train mode!")

            if is_training and (args.method != 'ProcessTrain' or args.loss == "LabelSmoothingSquaredLoss"):
                # print("You are inside backward function")
                if (args.model.startswith("bert") and args.use_bert_params): 
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                run_stats = loss_computer.get_stats(model, args)
                csv_logger.log(epoch, batch_idx, run_stats)

                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                if wandb is not None:
                    wandb_stats = {
                        wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                    }
                    wandb_stats["epoch"] = epoch
                    wandb_stats["batch_idx"] = batch_idx
                    wandb.log(wandb_stats)

        if run_name is not None:
            save_dir = "/".join(csv_logger.path.split("/")[:-1])
            if args.method != 'AUX2':
                output_df.to_csv(
                    os.path.join(save_dir, 
                                    f"output_{wandb_group}_epoch_{epoch}.csv"))
                print("Saved", os.path.join(save_dir, 
                                    f"output_{wandb_group}_epoch_{epoch}.csv"))


        if (not is_training) or loss_computer.batch_count > 0:
            run_stats = loss_computer.get_stats(model, args)
            if wandb is not None:
                assert wandb_group is not None
                wandb_stats = {
                    wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                }
                wandb_stats["epoch"] = epoch
                wandb_stats["batch_idx"] = batch_idx
                wandb.log(wandb_stats)
                print("logged to wandb")

            csv_logger.log(epoch, batch_idx, run_stats)
            csv_logger.flush()
            spurious_score, weighted_spurious_score, curr_worst_group_acc, class_difference, avg_acc = loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
            return spurious_score, weighted_spurious_score, curr_worst_group_acc, class_difference, avg_acc

def train(
    model,
    criterion,
    dataset,
    logger,
    train_csv_logger,
    val_csv_logger,
    test_csv_logger,
    args,
    epoch_offset,
    process_training = False, 
    csv_name=None,
    wandb=None,
):
    

    if args.method == 'AUX1':
        if args.dataset == 'MultiNLI':
            model.b = torch.nn.parameter.Parameter(torch.zeros(len(dataset['train_data'])+len(dataset['val_data'])+len(dataset['test_data']), 3)) # CUB 4795 train, 11788 total
        else:
            model.b = torch.nn.parameter.Parameter(torch.zeros(len(dataset['train_data'])+len(dataset['val_data'])+len(dataset['test_data']), 2)) # CUB 4795 train, 11788 total
    
    model = model.to(device)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, dataset["train_data"].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset["train_data"].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=dataset["train_data"],
        dataset_name=args.dataset,
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
        joint_dro_alpha=args.joint_dro_alpha,
    )

    # BERT uses its own scheduler and optimizer
    if (args.model.startswith("bert") and args.use_bert_params): 
        print("We are using AdamW!")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr,
                          eps=args.adam_epsilon)
        t_total = len(dataset["train_loader"]) * args.n_epochs
        print(f"\nt_total is {t_total}\n")
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=args.warmup_steps,
                                         t_total=t_total)
    else:
        print("We are using SGD!")
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            # weight_decay=0.2, #l2 regularization
        )
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08,
            )
        else:
            scheduler = None
    

    best_val_acc = 0
    # args.n_epoch = 13
    # print("args.n_epoch: ", args.n_epoch)
    # for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
    epoch = epoch_offset
    if args.method == 'AUX1':
        best_epoch, spurious_score_history = 0, 0
        best_weighted_epoch, weighted_spurious_score_history = 0, 0

        best_test_epoch, test_spurious_score_history = 0, 0
        best_test_weighted_epoch, test_weighted_spurious_score_history = 0, 0

    
    if args.method == 'AUX2':
        best_worst_group_acc_val = 0.0 
        best_worst_group_epoch_val = 0

        best_class_diff = 100000
        best_class_diff_epoch = 0

        best_average_acc = 0
        best_average_acc_epoch = 0

        robust_validation_worst_correponding_test_worst = 0 
        robust_validation_worst_correponding_test_average = 0 
        robust_validation_worst_epoch = 0

        smallest_class_difference_corresponding_test_worst = 0
        smallest_class_difference_corresponding_test_avg = 0
        smallest_class_difference_corresponding_epoch = 0

        class_one_acc = []
        class_two_acc = []
        if args.dataset == "MultiNLI":
            class_third_acc = []

    while True:

        logger.write("\nEpoch [%d]:\n" % epoch)
        logger.write(f"Training:\n")
        run_epoch(
            epoch,
            model,
            optimizer,
            dataset["train_loader"],
            train_loss_computer,
            logger,
            train_csv_logger,
            args,
            is_training=True,
            process_training=process_training,
            csv_name=csv_name,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            wandb_group="train",
            wandb=wandb,
        )

        logger.write(f"\nValidation:\n")
        val_loss_computer =  LossComputer(
            criterion,
            loss_type=args.loss_type,
            dataset=dataset["val_data"],
            dataset_name=args.dataset, 
            alpha=args.alpha,
            gamma=args.gamma,
            adj=adjustments,
            step_size=args.robust_step_size,
            normalize_loss=args.use_normalized_loss,
            btl=args.btl,
            min_var_weight=args.minimum_variational_weight,
            joint_dro_alpha=args.joint_dro_alpha,
        )
        spurious_score_cur, weighted_score_cur, curr_worst_group_acc_val, curr_class_difference, curr_average_acc = run_epoch(
            epoch,
            model,
            optimizer,
            dataset["val_loader"],
            val_loss_computer,
            logger,
            val_csv_logger,
            args,
            process_training=False, 
            is_training=False,
            csv_name=csv_name,
            wandb_group="val",
            wandb=wandb,
        )
       
        if args.method == 'AUX1':
            if spurious_score_cur > spurious_score_history:
                spurious_score_history = spurious_score_cur
                torch.save(model, os.path.join(args.log_dir,
                                            "AUX1_best_model.pth"))
                best_epoch = epoch
            if weighted_score_cur > weighted_spurious_score_history:
                weighted_spurious_score_history = weighted_score_cur
                torch.save(model, os.path.join(args.log_dir,
                                            "AUX1_best_weighted_model.pth"))
                best_weighted_epoch = epoch
            # logger.write(f'Current best spurious score epoch: {best_epoch}\n')
            # logger.write(f'Current best weighted spurious score epoch: {best_weighted_epoch}\n')
        

        if args.method == 'AUX2':
            if curr_worst_group_acc_val > best_worst_group_acc_val:
                best_worst_group_acc_val = curr_worst_group_acc_val
                best_worst_group_epoch_val = epoch 
                torch.save(model, os.path.join(args.log_dir,
                                                "AUX2_best_worst_group_val_model.pth"))
            if curr_class_difference < best_class_diff:
                best_class_diff = curr_class_difference
                best_class_diff_epoch = epoch
                torch.save(model, os.path.join(args.log_dir,
                                                "AUX2_smallest_class_diff_model.pth"))
            
            torch.save(model, os.path.join(args.log_dir,
                                                "AUX2_last_model.pth"))



        # Test set; don't print to avoid peeking
        if dataset["test_data"] is not None:
            test_loss_computer = LossComputer(
                criterion,
                loss_type=args.loss_type,
                dataset=dataset["test_data"],
                dataset_name=args.dataset,
                step_size=args.robust_step_size,
                alpha=args.alpha,
                gamma=args.gamma,
                adj=adjustments,
                normalize_loss=args.use_normalized_loss,
                btl=args.btl,
                min_var_weight=args.minimum_variational_weight,
                joint_dro_alpha=args.joint_dro_alpha,
            )
            
            spurious_score_cur, weighted_score_cur, curr_worst_test_group_acc, _, test_curr_avg_acc = run_epoch(
                epoch,
                model,
                optimizer,
                dataset["test_loader"],
                test_loss_computer,
                logger=None,
                csv_logger=test_csv_logger,
                args=args,
                process_training=False,
                is_training=False,
                csv_name=csv_name,
                wandb_group="test",
                wandb=wandb,
            )

            if args.method == 'AUX1':
                # print("args.method: ", args.method)
                # print("You are at method AUX1!")
                if spurious_score_cur > test_spurious_score_history:
                    test_spurious_score_history = spurious_score_cur
                    torch.save(model, os.path.join(args.log_dir,
                                                "AUX1_best_test_model.pth"))
                    best_test_epoch = epoch
                if weighted_score_cur > test_weighted_spurious_score_history:
                    test_weighted_spurious_score_history = weighted_score_cur
                    torch.save(model, os.path.join(args.log_dir,
                                                "AUX1_best_weighted_test_model.pth"))
                    best_test_weighted_epoch = epoch
        

            if args.method == 'AUX2':
                
                if epoch == best_worst_group_epoch_val:
                    robust_validation_worst_correponding_test_worst = curr_worst_test_group_acc
                    robust_validation_worst_correponding_test_average = test_curr_avg_acc
                    robust_validation_worst_epoch = epoch 
                if epoch == best_class_diff_epoch:
                    smallest_class_difference_corresponding_test_worst = curr_worst_test_group_acc
                    smallest_class_difference_corresponding_test_avg = test_curr_avg_acc
                    smallest_class_difference_corresponding_epoch = epoch 
                logger.write(f'\nBest Stats Info:\n')
                logger.write(f'Robust Best Validation Worst-group Accuracy: {best_worst_group_acc_val}\n')
                logger.write(f'Robust Validation Worst-group Corresponding Test Wort-group Acc: {robust_validation_worst_correponding_test_worst}\n')
                logger.write(f'Robust Validation Worst-group Corresponding Test Average Acc: {robust_validation_worst_correponding_test_average}\n')
                logger.write(f'Robust Validation Worst-group Corresponding Epoch: {robust_validation_worst_epoch}\n')
                logger.write(f'Smallest Validation class difference: {best_class_diff}\n')
                logger.write(f'Smallest class different Corresponding Test Worst-group Acc: {smallest_class_difference_corresponding_test_worst}\n')
                logger.write(f'Smallest class different Corresponding Test Average Acc: {smallest_class_difference_corresponding_test_avg}\n')
                logger.write(f'Smallest class different Corresponding Epoch {smallest_class_difference_corresponding_epoch}\n')

                


        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                logger.write("Current lr: %f\n" % curr_lr)

        if args.scheduler and args.model != "bert":
            if args.loss_type == "group_dro":
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss,
                    val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(
                val_loss)  # scheduler step to update lr at the end of epoch
        
        if args.method == 'AUX1': # and (args.dataset == "jigsaw" or args.dataset == "MultiNLI"):
            # Only save entire model at AUX1
            print(f"Epoch {epoch} model saved!")
            torch.save(model, os.path.join(args.log_dir,
                                            "%d_model.pth" % epoch))
        
        if args.method == 'ProcessTrain' and args.loss == "LabelSmoothingSquaredLoss":
            print(f"Epoch {epoch} model saved LabelSmoothingSquaredLoss!")
            torch.save(model, os.path.join(args.log_dir,
                                           "labelSmoothing_SqauredLoss_%d_model.pth" % epoch))
        if epoch % args.save_step == 0 and args.method == 'AUX1':
            torch.save(model, os.path.join(args.log_dir,
                                           "%d_model.pth" % epoch))
        
        # if args.method == "AUX2":
        #     if curr_val_acc > best_val_acc:
        #         torch.save(model, os.path.join(args.log_dir, "best_val_acc_model.pth"))
        #     if 
        # if args.save_last and args.method == 'ProcessTrain':
        #     torch.save(model, os.path.join(args.log_dir, "last_model.pth"))

        # if args.method == "ProcessTrain" and args.loss == "LabelSmoothingSquaredLoss":
        #     torch.save(model, os.path.join(args.log_dir, "%d_LabelSmoothingSquaredLoss"))

        if args.save_best and args.method == 'AUX1':
            if args.loss_type == "group_dro" or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f"Current validation accuracy: {curr_val_acc}\n")
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, "best_model.pth"))
                logger.write(f"Best model saved at epoch {epoch}\n")

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(
                train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write("Adjustments updated\n")
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f"  {train_loss_computer.get_group_name(group_idx)}:\t"
                    f"adj = {train_loss_computer.adj[group_idx]:.3f}\n")
        logger.write("\n")

        epoch += 1

        if epoch >= epoch_offset + args.n_epochs:
            break
    
        if args.method == 'AUX1':
            with open(os.path.join(args.log_dir, 'best_epoch.txt'), 'w') as file:
                file.write(str(best_epoch))
            with open(os.path.join(args.log_dir, 'best_weighted_epoch.txt'), 'w') as file:
                file.write(str(best_weighted_epoch))
            with open(os.path.join(args.log_dir, 'best_test_epoch.txt'), 'w') as file:
                file.write(str(best_test_epoch))
            with open(os.path.join(args.log_dir, 'best_test_weighted_epoch.txt'), 'w') as file:
                file.write(str(best_test_weighted_epoch))
            with open(os.path.join(args.log_dir, 'information.txt'), 'w') as file:
                file.write(f"lr {args.lr}\n")
                file.write(f"batch_size {args.batch_size}\n")
                file.write(f"dataset {args.dataset}\n")
                file.write(f"log_dir {args.log_dir}\n")
                file.write(f"loss_type {args.loss}\n")
                file.write(f"n_epochs {args.n_epochs}\n")
                file.write(f"aux_lambda {args.aux_lambda}\n")
                file.write(f"seed {args.seed}\n")