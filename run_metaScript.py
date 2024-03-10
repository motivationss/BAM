import subprocess 
import os 
import argparse
import pandas as pd 
import numpy as np 
from scipy.special import softmax
import time 
import torch 

dataset_dir = {
    "CUB": "cub/data/waterbird_complete95_forest2water2/",
    "CelebA": "celebA/data/",
    "MultiNLI": "multinli/data/",
    "jigsaw": "jigsaw/data/",
}


def sub_args_input(input_str):
    # str is "A B C D" where A B C D are input words separated by " "
    return input_str.split(" ")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CUB', choices=['CUB', 'jigsaw', "CelebA", "MultiNLI"])
    parser.add_argument("--aux_lambda", type=str, default='0.5')
    parser.add_argument("--stageOne_lr", type=str, default="1e-5")
    parser.add_argument("--stageTwo_lr", type=str, default="1e-5")
    parser.add_argument("--stageOne_wd", type=float, default=0)
    parser.add_argument("--stageTwo_wd", type=float, default=1e-5)

    parser.add_argument("--up_weights", type=int, nargs='*')
    parser.add_argument("--subsample_non_error", type=float, nargs='*', default=None) 
    parser.add_argument("--seed", type=str, default='0')
    parser.add_argument("--stageOne_epochs", type=int, default=21)
    parser.add_argument("--stageTwo_epochs", type=int, default=30)
    parser.add_argument("--stageOne_T", type=int, nargs='*', help="T")

    parser.add_argument("--use_weighted_spurious_score",default=False, action="store_true",
                        help="heuristics for helping choosing T")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--ProcessWhole", default=False, action="store_true", help="if process the whole Stage 1")
    parser.add_argument("--load_new_model", default=False, action="store_true", help="Two-M/One-M")



    parser.add_argument("--start_from_StageTwo", default=False, action="store_true", help="When Stage One has been runned")    

    args = parser.parse_args()

    device = torch.device('cuda')


    ################################## Step 1 ###########################################

    
    if args.dataset == "CUB":
        args.stageOne_lr = "1e-5" 
        args.stageTwo_lr = "1e-5" 
        args.stageOne_wd = 0 
        args.stageTwo_wd = 1.0 
    elif args.dataset == "CelebA":
        args.stageOne_lr = 1e-5 
        args.stageTwo_lr = 1e-5 
        args.stageOne_wd = 0 
        args.stageTwo_wd = 0.1
    elif args.dataset == "MultiNLI":
        args.stageOne_lr = 2e-5 
        args.stageTwo_lr = 1e-5 
        args.stageOne_wd = 0 
        args.stageTwo_wd = 0.1
    elif args.dataset == "jigsaw":
        args.stageOne_lr = 2e-5 
        args.stageTwo_lr = 1e-5 
        args.stageOne_wd = 0 
        args.stageTwo_wd = 0.01

    aux_lambda = args.aux_lambda
    dataset = args.dataset 
    nepochs_Stage1 = args.stageOne_epochs 
    nepochs_Stage2 = args.stageTwo_epochs
    lr_Stage1 = args.stageOne_lr
    lr_Stage2 = args.stageTwo_lr
    wd_Stage1 = args.stageOne_wd
    wd_Stage2 = args.stageTwo_wd
    up_weights = args.up_weights
    subsample_non_error = args.subsample_non_error if args.subsample_non_error else [1.0]
    stageOne_T = args.stageOne_T
    seed = args.seed

    if args.dataset == "CUB":
        root_dir = "./cub"
        batch_size = 64 
        exp_name = 'CUB_sample_exp'
        target = "waterbird_complete95"
        confounder_name = "forest2water2"
        model = "resnet50"
        metadata_csv_name = "metadata.csv"
        metadata_path = "./cub/data/waterbird_complete95_forest2water2/metadata.csv"
    elif args.dataset == "CelebA":
        root_dir = "./"
        batch_size = 128 
        exp_name = 'CelebA_sample_exp'
        target = "Blond_Hair"
        confounder_name = "Male"
        model = "resnet50"
        metadata_csv_name = "metadata.csv"
        metadata_path = "./celebA/data/metadata.csv"
    elif args.dataset == "MultiNLI":
        root_dir = "./"
        batch_size = 32 
        exp_name = 'MultiNLI_sample_exp'
        target = "gold_label_random"
        confounder_name = "sentence2_has_negation"
        model = "bert"
        metadata_csv_name = "metadata.csv"
        metadata_path = "./multinli/data/metadata.csv"
    elif args.dataset == "jigsaw":
        root_dir = "./jigsaw"
        batch_size = 16 
        exp_name = 'jigsaw_sample_exp'
        target = "toxicity"
        confounder_name = "identity_any"
        batch_size = 16 # no-bert-param: 24
        model = "bert-base-uncased"
        metadata_csv_name = "all_data_with_identities.csv"
        metadata_path = "./jigsaw/data/all_data_with_identities.csv"
    else:
        raise NotImplementedError("The dataset is not Implemented, plese check the code!")
    
    exp_dir = os.path.join('result', dataset, exp_name)

    # Step 1 First Train: 
    # step 1 no upweight
    
    stage_1_sub_exp_name = f"AUX1_epochs_{nepochs_Stage1}_lr_{lr_Stage1}_weightDecay_{wd_Stage1}_auxLambda_{aux_lambda}_seed_{seed}"
    folder_name = stage_1_sub_exp_name # AKA folder name
    stage_1_sub_exp_dir = os.path.join(exp_dir, stage_1_sub_exp_name) 
    stage_1_training_out_dir = os.path.join(stage_1_sub_exp_dir, "model_outputs") 

    if not os.path.exists(stage_1_training_out_dir):
        os.makedirs(stage_1_training_out_dir) 
    
    Stage1_firstTrain_script = f"python run_expt.py -s confounder -d {dataset} -t {target} -c {confounder_name}" \
    + f" --batch_size {batch_size} --root_dir {root_dir} --n_epochs {nepochs_Stage1}" \
    + f" --aug_col wrong_1_times --log_dir {stage_1_training_out_dir}" \
    + f" --lr {lr_Stage1} --weight_decay {wd_Stage1} --up_weight 0" \
    + f" --metadata_csv_name {metadata_csv_name} --model {model} --use_bert_params 0 --method AUX1" \
    + f" --aux_lambda {aux_lambda} --seed {args.seed}" 

    
    if not args.start_from_StageTwo:
        torch.cuda.reset_peak_memory_stats(device)
        print("Stage 1 First Train Script: ", Stage1_firstTrain_script)
        print("###################### Stage 1 Training #############################")
        subprocess.call(sub_args_input(Stage1_firstTrain_script))
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        print(f'Current Max memory allocated on {device}: {max_memory_allocated / 1024 ** 2} MB')

    ################ Stage 1 Select Best Epoch and Process Train ########################
    # Choice one: Process Whole - Process Whole Dataset 
    # Choice two: Process Single Epoch T - best_epoch + process_training
    
    # print("stage1trainingout: ", stage_1_training_out_dir)
    # exit()
    
    if not args.start_from_StageTwo:
        # Get error set without b
        if args.ProcessWhole:
            for T in args.stageOne_epochs:
                Stage1_ProcessTrain_script = f"python run_expt.py -s confounder -d {dataset} -t {target} -c {confounder_name}" \
                + f" --batch_size {batch_size} --root_dir {root_dir} --n_epochs {nepochs_Stage1}" \
                + f" --aug_col wrong_1_times --log_dir {stage_1_training_out_dir}" \
                + f" --lr {lr_Stage1} --weight_decay {wd_Stage1} --up_weight 0" \
                + f" --metadata_csv_name {metadata_csv_name} --model {model} --use_bert_params 0 --method ProcessTrain" \
                + f" --aux_lambda {aux_lambda} --seed {args.seed} --loss CrossEntropy" \
                + f" --log_dir_old {stage_1_sub_exp_dir}" \
                + (f" --ProcessWhole" if args.ProcessWhole else "") \
                + (f" --process_training --best_epoch {T} --loadModel {T}" if not args.ProcessWhole else "")
                
                print("Stage 1 Process Train Script: ", Stage1_ProcessTrain_script)
                subprocess.call(sub_args_input(Stage1_ProcessTrain_script))
        
        else: 
            for T in stageOne_T:
                Stage1_ProcessTrain_script = f"python run_expt.py -s confounder -d {dataset} -t {target} -c {confounder_name}" \
                + f" --batch_size {batch_size} --root_dir {root_dir} --n_epochs {nepochs_Stage1}" \
                + f" --aug_col wrong_1_times --log_dir {stage_1_training_out_dir}" \
                + f" --lr {lr_Stage1} --weight_decay {wd_Stage1} --up_weight 0" \
                + f" --metadata_csv_name {metadata_csv_name} --model {model} --use_bert_params 0 --method ProcessTrain" \
                + f" --aux_lambda {aux_lambda} --seed {args.seed} --loss CrossEntropy" \
                + f" --log_dir_old {stage_1_sub_exp_dir}" \
                + (f" --ProcessWhole" if args.ProcessWhole else "") \
                + (f" --process_training --best_epoch {T} --loadModel {T}" if not args.ProcessWhole else "")
                
                # if not args.start_from_StageTwo:
                print("Stage 1 Process Train Script: ", Stage1_ProcessTrain_script)
                subprocess.call(sub_args_input(Stage1_ProcessTrain_script))
            
        
            

    
    ############################# Error Set Generation ##################################

    # Simple Heuristics 
    if args.use_weighted_spurious_score:
        with open(os.path.join(stage_1_training_out_dir, 'best_test_weighted_epoch.txt'), 'r') as file:
            final_epoch = int(float(file.read()))
    else:
        with open(os.path.join(stage_1_training_out_dir, 'best_test_epoch.txt'), 'r') as file:
            final_epoch = int(float(file.read()))
    
    best_epoch_info = final_epoch
    info_dict = {}
    with open(os.path.join(stage_1_training_out_dir, 'information.txt'), 'r') as f:
        for line in f:
            (k, v) = line.split()
            info_dict[k] = v
    print(info_dict)

    
    if stageOne_T is not None:
        processWhole_stage2 = False
    elif args.ProcessWhole:
        processWhole_stage2 = True
    else:
        raise NotImplementedError("Specify stageOne_T or ProcessWhole")
    
    num_processes = info_dict["n_epochs"] if processWhole_stage2 else len(stageOne_T) 
    # Process All Epochs or just specified "T"s
    # num_processes = args.total_epochs_evaluate if args.total_epochs_evaluate is not None else num_processes 

    for final_epoch in range(int(num_processes)):
        
        current_T = stageOne_T[final_epoch]
        final_epoch = final_epoch if processWhole_stage2 else current_T # Process correct error set
        loadModel = final_epoch if processWhole_stage2 else current_T # Either traverse or T
        if args.load_new_model:
            loadModel = None # This recovers Two-M

        train_df = pd.read_csv(os.path.join(stage_1_training_out_dir, f"output_train_epoch_{final_epoch}.csv"))
        train_df = train_df.sort_values(f"indices_None_epoch_{final_epoch}_val")
        train_df["wrong_1_times"] = (1.0 * (train_df[f"y_pred_None_epoch_{final_epoch}_val"] != train_df[f"y_true_None_epoch_{final_epoch}_val"])).apply(np.int64)
        print("Total wrong", np.sum(train_df['wrong_1_times']), "Total points", len(train_df))

        original_df = pd.read_csv(metadata_path)
        # Bug fixed from JTT
        if dataset == "jigsaw":
            original_train_df = original_df[original_df["split"] == "train"]
        else:
            original_train_df = original_df[original_df["split"] == 0]
        
        if dataset == "jigsaw" or dataset == "MultiNLI":
            original_train_df = original_train_df.drop(['Unnamed: 0'], axis=1)

        merged_csv = original_train_df.join(train_df.set_index(f"indices_None_epoch_{final_epoch}_val"))

        if dataset == "CUB":
            merged_csv["spurious"] = merged_csv['y'] != merged_csv["place"]
        elif dataset == "CelebA":
            merged_csv = merged_csv.replace(-1, 0)
            assert 0 == np.sum(merged_csv[merged_csv["split"] == 0]["Blond_Hair"] != merged_csv[merged_csv["split"] == 0][f"y_true_None_epoch_{final_epoch}_val"])
            merged_csv["spurious"] = (merged_csv["Blond_Hair"] == merged_csv["Male"]) 
        elif dataset == "jigsaw":
            merged_csv["spurious"] = original_train_df["toxicity"] >= 0.5
            print("merged_csv len of toxicity: ", len(merged_csv["toxicity"]))
        elif dataset == "MultiNLI":
            # merged_csv["spurious"] = (
            #         (merged_csv["gold_label"] == 0)
            #         & (merged_csv["sentence2_has_negation"] == 0)
            #     ) | (
            #         (merged_csv["gold_label"] == 1)
            #         & (merged_csv["sentence2_has_negation"] == 1)
            #     )
            merged_csv["spurious"] = (
                    (merged_csv["gold_label"] == 2)
                    & (merged_csv["sentence2_has_negation"] == 1)
                ) | (
                    (merged_csv["gold_label"] == 1)
                    & (merged_csv["sentence2_has_negation"] == 1)
                )
        else: 
            raise NotImplementedError
        
        print("Number of spurious", np.sum(merged_csv['spurious']))
    
        # Make columns for our spurious and our nonspurious
        merged_csv["our_spurious"] = merged_csv["spurious"] & merged_csv["wrong_1_times"]
        merged_csv["our_nonspurious"] = (merged_csv["spurious"] == 0) & merged_csv["wrong_1_times"]
        print("Number of our spurious: ", np.sum(merged_csv["our_spurious"]))
        print("Number of our nonspurious:", np.sum(merged_csv["our_nonspurious"]))

        if dataset == "MultiNLI":
            print("\nDetailed Error Set Information: ")
            print("gold_label_random = 0, sentence2_has_negation = 0: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 0)
            & (merged_csv["sentence2_has_negation"] == 0)))

            print("gold_label_random = 0, sentence2_has_negation = 1: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 0)
            & (merged_csv["sentence2_has_negation"] == 1)))

            print("gold_label_random = 1, sentence2_has_negation = 0: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 1)
            & (merged_csv["sentence2_has_negation"] == 0)))

            print("gold_label_random = 1, sentence2_has_negation = 1: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 1)
            & (merged_csv["sentence2_has_negation"] == 1)))

            print("gold_label_random = 2, sentence2_has_negation = 0: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 2)
            & (merged_csv["sentence2_has_negation"] == 0)))

            print("gold_label_random = 2, sentence2_has_negation = 1: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 2)
            & (merged_csv["sentence2_has_negation"] == 1)))
        elif dataset == "CUB":
            print("Detailed Error Set Information: ")
            num_00 = np.sum((merged_csv['y'] == 0) & (merged_csv['place'] == 0)
            & (merged_csv["wrong_1_times"]))
            num_01 = np.sum((merged_csv['y'] == 0) & (merged_csv['place'] == 1)
            & (merged_csv["wrong_1_times"]))
            num_10 = np.sum((merged_csv['y'] == 1) & (merged_csv['place'] == 0)
            & (merged_csv["wrong_1_times"]))
            num_11 = np.sum((merged_csv['y'] == 1) & (merged_csv['place'] == 1)
            & (merged_csv["wrong_1_times"]))
            print(f"Waterbird in Water: {num_00}")
            print(f"Waterbird in Land: {num_01}")
            print(f"Landbird in Water: {num_10}")
            print(f"Landbird in Land: {num_11}")
        elif dataset == "jigsaw":
            print("\nDetailed Error Set Information: ")
            num_00 = np.sum((merged_csv['identity_any'] == 0) & (merged_csv["toxicity"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_01 = np.sum((merged_csv['identity_any'] == 1) & (merged_csv["toxicity"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_10 = np.sum((merged_csv['identity_any'] == 0) & (merged_csv["toxicity"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            num_11 = np.sum((merged_csv['identity_any'] == 1) & (merged_csv["toxicity"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            print(f"Not-toxic No identity: {num_00}")
            print(f"Not-toxic identity: {num_01}")
            print(f"Toxic No identity: {num_10}")
            print(f"Not-toxic Identity: {num_11}")
        elif dataset == "CelebA":
            print("\nDetailed Error Set Information: ")
            num_00 = np.sum((merged_csv['Blond_Hair'] == 0) & (merged_csv["Male"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_01 = np.sum((merged_csv['Blond_Hair'] == 1) & (merged_csv["Male"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_10 = np.sum((merged_csv['Blond_Hair'] == 0) & (merged_csv["Male"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            num_11 = np.sum((merged_csv['Blond_Hair'] == 1) & (merged_csv["Male"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            print(f"Not-blond Female: {num_00}")
            print(f"Blond Female: {num_01}")
            print(f"Not-blond Male: {num_10}")
            print(f"Blond Male: {num_11}")
        
        train_probs_df= merged_csv.fillna(0)

         
        # Find confidence (just in case doing threshold)
        if dataset == "MultiNLI":
            probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1", f"pred_prob_None_epoch_{final_epoch}_val_2"]]), axis = 1)
            train_probs_df["probs_0"] = probs[:,0]
            train_probs_df["probs_1"] = probs[:,1]
            train_probs_df["probs_2"] = probs[:,2]
            train_probs_df["confidence"] = (train_probs_df['gold_label']==0) * train_probs_df["probs_0"] + (train_probs_df['gold_label']==1) * train_probs_df["probs_1"] + (train_probs_df['gold_label']==2) * train_probs_df["probs_2"]
        else:
            probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1"]]), axis = 1)
            train_probs_df["probs_0"] = probs[:,0]
            train_probs_df["probs_1"] = probs[:,1]
            if dataset == 'CelebA':
                train_probs_df["confidence"] = train_probs_df["Blond_Hair"] * train_probs_df["probs_1"] + (1 - train_probs_df["Blond_Hair"]) * train_probs_df["probs_0"]
            elif dataset == 'CUB':
                train_probs_df["confidence"] = train_probs_df["y"] * train_probs_df["probs_1"] + (1 - train_probs_df["y"]) * train_probs_df["probs_0"]
            elif dataset == 'jigsaw':
                train_probs_df["confidence"] = (train_probs_df["toxicity"] >= 0.5) * train_probs_df["probs_1"] + (train_probs_df["toxicity"] < 0.5)  * train_probs_df["probs_0"]
        
        train_probs_df[f"confidence_thres{args.conf_threshold}"] = (train_probs_df["confidence"] < args.conf_threshold).apply(np.int64)
        if dataset == 'CelebA':
            assert(np.sum(train_probs_df[f"confidence_thres{args.conf_threshold}"] != train_probs_df["wrong_1_times"]) == 0)
        
        # Save csv into new dir for the run, and generate downstream runs
        if not os.path.exists(f"{exp_dir}/train_downstream_{folder_name}/final_epoch{final_epoch}"):
            os.makedirs(f"{exp_dir}/train_downstream_{folder_name}/final_epoch{final_epoch}")
        root = f"{exp_dir}/train_downstream_{folder_name}/final_epoch{final_epoch}"

        ### IMPORTANT #### 
        stage_2_metadata_path = f"{root}/metadata_aug.csv"
        train_probs_df.to_csv(stage_2_metadata_path)

        # Process Whole
        # if args.ProcessWhole:
        meta_aug_csv_root = f"{exp_dir}/train_downstream_{folder_name}"
        if not os.path.exists(f"{root}/metadata_aug_files"):
            os.makedirs(f"{root}/metadata_aug_files")
        if not os.path.exists(f"{meta_aug_csv_root}/metadata_aug_files"):
            os.makedirs(f"{meta_aug_csv_root}/metadata_aug_files")
        loss_type = info_dict["loss_type"]
        b_input = "withoutb" #if args.withoutb else "withb"
        train_probs_df.to_csv(f"{meta_aug_csv_root}/metadata_aug_files/metadata_aug_epoch{final_epoch}_{b_input}_{loss_type}.csv")
        train_probs_df.to_csv(f"{root}/metadata_aug_files/metadata_aug_epoch{final_epoch}_{b_input}_{loss_type}.csv")
        root = f"train_downstream_{folder_name}/final_epoch{final_epoch}"
        print(f"Epoch {final_epoch} save!")
        print("\n"*3)


        # Generate Stage 2 Folders 
        ################################################################################
        ############ Add Modification here for more ablations in Stage 2 ###############
        ################################################################################

        for up_weight in up_weights:
            for subsample_nonerr_ratio in subsample_non_error:
                print(f"############################ STAGE 2 upweight {up_weight} T {loadModel} ####################")
                stage_2_sub_exp_name = f"AUX2_upweight_{up_weight}_epochs_{nepochs_Stage2}_lr_{lr_Stage2}_weightDecay_{wd_Stage2}_subsample_{subsample_nonerr_ratio}"
                
                stage_2_sub_exp_dir = os.path.join(exp_dir, root, stage_2_sub_exp_name)
                job_script_path = os.path.join(stage_2_sub_exp_dir, 'job.sh')

                stage_2_training_output_dir = os.path.join(stage_2_sub_exp_dir, "model_outputs")
                

                if not os.path.exists(stage_2_training_output_dir):
                    os.makedirs(stage_2_training_output_dir)
                
                Stage2_RobustTrain_script = f"python run_expt.py -s confounder -d {dataset} -t {target} -c {confounder_name}" \
                + f" --batch_size {batch_size} --root_dir {root_dir} --n_epochs {nepochs_Stage2}" \
                + f" --aug_col wrong_1_times --log_dir {stage_2_training_output_dir}" \
                + f" --metadata_path {stage_2_metadata_path}" \
                + f" --lr {lr_Stage2} --weight_decay {wd_Stage2} --up_weight {up_weight}" \
                + f" --subsample_propotion {subsample_nonerr_ratio}" \
                + f" --metadata_csv_name {metadata_csv_name} --model {model} --use_bert_params 1 --method AUX2" \
                + f" --seed {args.seed} --loss CrossEntropy" \
                + f" --log_dir_old {stage_1_sub_exp_dir}" \
                + (f" --loadModel {loadModel}" if loadModel is not None else "") \
                + (f" --load_new_model" if args.load_new_model else "")

                with open(job_script_path, 'w') as f:
                    f.write(Stage2_RobustTrain_script) 
                
                subprocess.call(sub_args_input(Stage2_RobustTrain_script))
    