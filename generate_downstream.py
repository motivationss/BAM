import argparse
import pandas as pd
import os
import subprocess

join = os.path.join

initialization_text = """"""

final_text = """

# done
echo "Done"
"""

dataset_dir = {
    "CUB": "cub/data/waterbird_complete95_forest2water2/",
    "CelebA": "celebA/data/",
    "MultiNLI": "multinli/data/",
    "jigsaw": "jigsaw/data/",
}

def get_spurious_col_csv(args):
    metadata_dir = join(join(args.results_dir, args.dataset), args.exp_name)
    # output_dir: results/dataset/exp_name/
    output_dir = metadata_dir
    # output_path: results/dataset/exp_name/metadata_aug.csv
    args.output_path = join(output_dir, args.output_csv_name)
    args.metadata_path = os.path.join(
            dataset_dir[args.dataset], args.metadata_csv_name
        )
    new_metadata = pd.read_csv(args.metadata_path)
    split_name = "split"

    if args.dataset == "jigsaw":
        train_data = new_metadata[new_metadata[split_name] == "train"]
    else:
        train_data = new_metadata[new_metadata[split_name] == 0]

    index_col = "Unnamed: 0"
    if args.dataset == "CUB":
        train_data["spurious"] = train_data["y"] != train_data["place"]
        index_col = "img_id"
    elif args.dataset == "CelebA":
        train_data["spurious"] = (
            (train_data["Blond_Hair"] == 1) & (train_data["Male"] == 1)
        ) | ((train_data["Blond_Hair"] == -1) & (train_data["Male"] == -1))

    elif args.dataset == "MultiNLI":
        train_data["spurious"] = (
            (train_data["gold_label"] == 0)
            & (train_data["sentence2_has_negation"] == 0)
        ) | (
            (train_data["gold_label"] == 1)
            & (train_data["sentence2_has_negation"] == 1)
        )
        true = "gold_label"
        
    spur_col = train_data[["spurious", index_col]]
    new_metadata = pd.merge(
        new_metadata, spur_col, how="outer", on=index_col
    )
    new_metadata = new_metadata.fillna(False)

    # Save metadata
    new_metadata.to_csv(args.output_path)
    

def generate_downstream_commands(args):
    exp_dir = join(join(args.results_dir, args.dataset), args.exp_name)
    print("expdir: ", exp_dir)
    print("result_dir: ", args.results_dir)
    print("args.dataset: ", args.dataset)
    print("args.exp_name: ", args.exp_name)
    print("\n"*2)
    if args.metadata_path is None:
        metadata_path = join(exp_dir, args.csv_name)
    else:
        metadata_path = args.metadata_path
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if args.method == "all":
        methods = [
            "JTT",
            "AUX1",
            "AUX2",
            "GROUP_DRO",
            "ERM",
            "UPSAMPLE_TRUE_POINTS",
            "JOINT_DRO",
        ]
    else:
        methods = [args.method]
 
    for method in methods:
        best_epoch = None
        if method == "JTT":
            up_weights = [20, 50, 100] 
            loss_type = "erm"
            aug_col = args.aug_col
            confounder_name = args.confounder_name
        elif method == 'AUX2':
            up_weights = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200] if args.dataset == "CUB" or args.dataset == "CelebA" else [4, 5, 6, 7]
            lrs = [1e-4, 1e-5, 1e-6] if args.dataset == "MultiNLI" else [args.lr]
            weight_decays = [1e-4, 1e-2, 1e-1] if args.dataset == "MultiNLI" else [args.weight_decay]
            loss_type = 'erm'
            aug_col = args.aug_col
            confounder_name = args.confounder_name
        elif method == 'AUX1':
            up_weights = [0]
            lrs = [args.lr]
            weight_decays = [args.weight_decay]
            loss_type = 'erm'
            aug_col = args.aug_col
            confounder_name = args.confounder_name
            if args.dataset == "jigsaw":
                args.use_bert_params = 0
        elif method == 'ProcessTrain':
            lrs = [args.lr]
            weight_decays = [args.weight_decay]
            up_weights = [0]
            loss_type = 'erm'
            aug_col = args.aug_col
            confounder_name = args.confounder_name

            if args.use_weighted_spurious_score:
                with open(os.path.join(args.log_dir_old, 'model_outputs/best_test_weighted_epoch.txt'), 'r') as file:
                    best_epoch = int(float(file.read()))
            else:
                with open(os.path.join(args.log_dir_old, 'model_outputs/best_test_epoch.txt'), 'r') as file:
                    best_epoch = int(float(file.read()))
            

        elif method == "ERM":
            up_weights = [0]
            loss_type = "erm"
            aug_col = None
            confounder_name = args.confounder_name
            args.use_bert_params = 0
        elif method == "GROUP_DRO":
            up_weights = [0]
            loss_type = "group_dro"
            aug_col = None
            confounder_name = args.confounder_name
        elif method == "UPSAMPLE_TRUE_POINTS":
            if args.dataset != 'jigsaw':
                get_spurious_col_csv(args)
            up_weights = [-1, 10, 50]
            loss_type = "erm"
            aug_col = "spurious"
            confounder_name = args.confounder_name
        elif "JOINT_DRO" in method:
            up_weights = [0]
            loss_type = "joint_dro"
            aug_col = None
            confounder_name = args.confounder_name
            joint_dro_alpha = float(method.split("_")[2].replace("a", ""))
        else:
            print(f"Unknown method {method}")
            continue
         
        for up_weight in up_weights:
            for lr in lrs:
                for weight_decay in weight_decays:

                    if args.method == 'AUX1' or args.method == 'ProcessTrain':
                        sub_exp_name = f"AUX1_upweight_{up_weight}_epochs_{args.n_epochs}_lr_{lr}_weight_decay_{weight_decay}_aux_lambda_{args.aux_lambda}"
                    else:
                        sub_exp_name = f"{method}_upweight_{up_weight}_epochs_{args.n_epochs}_lr_{lr}_weight_decay_{weight_decay}"
                    # sub_exp_dir: results/dataset/exp_name/sub_exp_name
                    sub_exp_dir = join(exp_dir, sub_exp_name)
                    # job_script_path: results/dataset/exp_name/sub_exp_name/job.sh
                    job_script_path = join(sub_exp_dir, args.job_script_name)
                    # job_script_path: results/dataset/exp_name/sub_exp_name/model_outputs/
                    training_output_dir = join(sub_exp_dir, "model_outputs")

                    if not os.path.exists(sub_exp_dir):
                        os.makedirs(sub_exp_dir)

                    with open(job_script_path, "w") as file:
                        file.write(
                            initialization_text.format(
                                args.memory, method + "_" + args.exp_name
                            )
                        )
                        print(
                            f"python /nfs/turbo/coe-vvh/jtt/run_expt.py -s confounder -d {args.dataset} -t {args.target} -c {confounder_name}"
                            + f" --batch_size {args.batch_size} --root_dir {args.root_dir} --n_epochs {args.n_epochs}"
                            + f" --aug_col {aug_col} --log_dir {training_output_dir}"
                            + f" --metadata_path {metadata_path}"
                            + f" --lr {lr} --weight_decay {weight_decay} --up_weight {up_weight}"
                            + f" --metadata_csv_name {args.metadata_csv_name}  --model {args.model} --use_bert_params {args.use_bert_params} --method {args.method}"
                            + (" --wandb" if not args.no_wandb else "")
                            + (f" --loss_type {loss_type}")
                            + (" --reweight_groups" if loss_type == "group_dro" else "")
                            + (f" --joint_dro_alpha {joint_dro_alpha}" if loss_type == "joint_dro" else "")
                            + (f" --aux_lambda {args.aux_lambda}" if args.aux_lambda is not None else "")
                            + (f" --log_dir_old {args.log_dir_old}" if args.log_dir_old is not None else "")
                            + (f" --resume" if args.method == "AUX2" else "")
                            + (f" --use_weighted_spurious_score" if args.use_weighted_spurious_score else "")
                            + (f" --process_training" if args.process_training else "")
                            + (f" --best_epoch {best_epoch}" if best_epoch is not None else "")
                            + (f" --seed {args.set_seed}" if args.set_seed is not None else "")
                            + (f" --loss {args.loss}")
                            + (f" --loadModel {args.loadModel}" if args.loadModel is not None else "")
                            + (f" --ProcessWhole" if args.ProcessWhole else "")
                            + (f" --load_new_model" if args.load_new_model else "")
                        )

                        file.write(
                            f"python /nfs/turbo/coe-vvh/jtt/run_expt.py -s confounder -d {args.dataset} -t {args.target} -c {confounder_name}"
                            + f" --batch_size {args.batch_size} --root_dir {args.root_dir} --n_epochs {args.n_epochs}"
                            + f" --aug_col {aug_col} --log_dir {training_output_dir} --metadata_path {metadata_path}"
                            + f" --lr {lr} --weight_decay {weight_decay} --up_weight {up_weight} --metadata_csv_name {args.metadata_csv_name} --model {args.model} --use_bert_params {args.use_bert_params} --method {args.method}"
                            + (" --wandb" if not args.no_wandb else "")
                            + (f" --loss_type {loss_type}")
                            + (" --reweight_groups" if loss_type == "group_dro" else "")
                            + (f" --joint_dro_alpha {joint_dro_alpha}" if loss_type == "joint_dro" else "")
                            + (f" --aux_lambda {args.aux_lambda}" if args.aux_lambda is not None else "")
                            + (f" --log_dir_old {args.log_dir_old}" if args.log_dir_old is not None else "")
                            + (f" --resume" if args.method == "AUX2" else "")
                            + (f" --use_weighted_spurious_score" if args.use_weighted_spurious_score else "")
                            + (f" --process_training" if args.process_training else "")
                            + (f" --best_epoch {best_epoch}" if best_epoch is not None else "")
                            + (f" --seed {args.set_seed}" if args.set_seed is not None else "")
                            + (f" --loss {args.loss}")
                            + (f" --loadModel {args.loadModel}" if args.loadModel is not None else "")
                            + (f" --ProcessWhole" if args.ProcessWhole else "")
                            + (f" --load_new_model" if args.load_new_model else "")
                        )
                        
                        file.write("\n")
                        file.write(final_text)
                    print(f"\nsaved in {job_script_path}\n\n")
        #             subprocess.run(f"sbatch {job_script_path}", check=True, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: automatically generate experiment name if nothing is provided.
    parser.add_argument("--dataset", type=str, default="CUB")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--method", type=str, default="JTT")

    # parser.add_argument("--up_weight", type=int, default=0)
    parser.add_argument('--aux_lambda', type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--use_bert_params", type=int, default=1)
    parser.add_argument(
        "--aug_col",
        type=str,
        default="wrong_1_times",
        help="column for upsampling (number of times to augment dataset by each point)",
    )
    parser.add_argument("--num_exps", type=int, default=0)
    parser.add_argument("--memory", type=int, default=None)
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="do not add wandb logging",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    # Default arguments (don't change)
    parser.add_argument("--csv_name",
                        type=str,
                        default="metadata_aug.csv")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument(
        "--job_script_name",
        type=str,
        default="job.sh",
        help="name for sbatch script for training models",
    )
    
    # Post Process args
    parser.add_argument(
        "--extension",
        type=str,
        default="0_None",
        help="extension on upstream exp csvs, only changes if upweight",
    )
    
    parser.add_argument(
        "--metadata_csv_name",
        type=str,
        default=None,
        help="name of the csv data file (dataset csv has to be placed in dataset folder).",
    )
    parser.add_argument(
        "--final_epoch",
        type=int,
        default=None,
        help="last epoch in training",
    )
    parser.add_argument(
        "--log_dir_old",
        type=str,
        default=None
    )
    # for intersection of error sets
    parser.add_argument("--set_seed", type=int, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    # AUX2 input
    parser.add_argument("--use_weighted_spurious_score", default=False, action="store_true")

    # Process training input
    parser.add_argument("--process_training", default=False, action="store_true")
    parser.add_argument("--loadModel", type=str, default=None)
    parser.add_argument("--ProcessWhole", default=False, action="store_true")
    # Loss
    parser.add_argument("--loss", type=str, default="CrossEntropy", 
    help="GCE, CrossEntropy, Sqaured Loss, LabelSmoothing")
    parser.add_argument("--load_new_model", default=False, action="store_true")
    args = parser.parse_args()

    args.output_csv_name = args.csv_name

    if args.exp_name is None:
        exp_dir = join(args.results_dir, args.dataset)
        experiments = os.listdir(exp_dir)
        assert (
            False
        ), f"Experiment name is required, here are the experiments:\n{experiments}"

    if args.dataset == "CUB":
        args.root_dir = "./cub"
        args.target = "waterbird_complete95"
        args.confounder_name = "forest2water2"
        args.model = "resnet50"
        args.batch_size = 64
        # args.n_epochs = 300
        # args.n_epochs = 100
        # args.n_epochs = 304 #aux 100
        # args.memory = 30 if not args.memory else args.memory
        args.metadata_csv_name = "metadata.csv" if not args.metadata_csv_name else args.metadata_csv_name
    elif args.dataset == "CelebA":
        args.root_dir = "./"
        args.target = "Blond_Hair"
        args.confounder_name = "Male"
        # args.n_epochs = 50
        args.model = "resnet50"
        # args.memory = 30 if not args.memory else args.memory
        args.metadata_csv_name = "metadata.csv" if not args.metadata_csv_name else args.metadata_csv_name
    elif args.dataset == "MultiNLI":
        args.root_dir = "./"
        args.target = "gold_label_random"
        args.confounder_name = "sentence2_has_negation"
#         args.lr = 0.00002
#         args.weight_decay = 0
        args.batch_size = 32
        args.model = "bert"
        # args.memory = 30 if not args.memory else args.memory
        # args.n_epochs = 5
        args.metadata_csv_name = "metadata.csv" if not args.metadata_csv_name else args.metadata_csv_name
    elif args.dataset == "jigsaw":
        args.root_dir = "./jigsaw"
        args.target = "toxicity"
        args.confounder_name = "identity_any"
#         args.lr = 1e-5 # no-bert-param: 2e-5
#         args.weight_decay = 0.01 # no-bert-param: 0.0
        args.batch_size = 16 # no-bert-param: 24
        # args.n_epochs = 3
        # args.n_epochs = 100
        args.model = "bert-base-uncased"
        # args.memory = 60 if not args.memory else args.memory
        args.final_epoch = 0
        args.metadata_csv_name = "all_data_with_identities.csv" if not args.metadata_csv_name else args.metadata_csv_name
    else:
        assert False, f"{args.dataset} is not a known dataset."

    generate_downstream_commands(args)