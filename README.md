# Bias Amplification Enhances Minority Performance

This code repository is adapted from [JTT's code](https://github.com/anniesch/jtt) and implements the paper "Bias Amplification Enhances Minority Group Performance". 

## Environment

Create an environment with the following commands:
```
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Downloading Datasets

- **Waterbirds:** Download waterbirds from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) and put it in `bam/cub`.
    - In that directory, our code expects `data/waterbird_complete95_forest2water2/` with `metadata.csv` inside.

- **CelebA:** Download CelebA from [here](https://www.kaggle.com/jessicali9530/celeba-dataset) and put it in `jtt/celebA`.
    - In that directory, our code expects the following files/folders:
        - data/list_eval_partition.csv
        - data/list_attr_celeba.csv
        - data/img_align_celeba/

- **MultiNLI:** Follow instructions [here](https://github.com/kohpangwei/group_DRO#multinli-with-annotated-negations) to download this dataset and put in `bam/multinli`
    - In that directory, our code expects the following files/folders:
        - data/metadata.csv
        - glue_data/MNLI/cached_dev_bert-base-uncased_128_mnli
        - glue_data/MNLI/cached_dev_bert-base-uncased_128_mnli-mm
        - glue_data/MNLI/cached_train_bert-base-uncased_128_mnli

- **CivilComments:** This dataset can be downloaded from [here](https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/) and put it in `bam/jigsaw`. In that directory, our code expects a folder `data` with the downloaded dataset.


## Sample Commands for running BAM

```
python run_metaScript.py --dataset CUB --aux_lambda 50 --stageOne_epoch 150 --stageOne_T 20 --stageTwo_epochs 150 --up_weights 140 --seed 42
```