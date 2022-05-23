# Striking a Balance: Alleviating Inconsistency in Pre-trained Models for Symmetric Classification Tasks

Source code for [ACL 2022 Findings](https://www.2022.aclweb.org/) paper: [Striking a Balance: Alleviating Inconsistency in Pre-trained Models for Symmetric Classification Tasks](https://aclanthology.org/2022.findings-acl.148.pdf)

<p align="center">
  <img align="center" src="https://github.com/ashutoshml/alleviating-inconsistency/blob/main/images/consistentbert.png" alt="Image" height="420" >
</p>

```diff
- [WIP] While most of the code should work, some scripts do need some amount of cleaning. Tentative date for fully-functional code: July 31, 2022.
```
## Dependencies

- compatible with python 3.9
- dependencies can be installed using `requirements.txt`
- model has been tested in multi-gpu setup also, please use `CUDA_VISIBLE_DEVICES=0,1,2,3` and `-n_gpus 4` incase of 4 gpus.

## Dataset

Download the datasets from the following url:

[Datasets](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/ashutosh_iisc_ac_in/EoOzEpz5gARBlJDPSs4RVnIBNAghWwPU0FG4cKYGzTpo9g?e=pzeqWx)

## Setup

To get the project's source code, clone the github repository:

```shell
$ git clone https://github.com/ashutoshml/alleviating-inconsistency.git
```

Install Conda: 

[Conda Installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Create and activate your virtual environment:

```shell
$ conda create -n venv python=3.9
$ conda activate venv
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

## Training the base models on qqp, paws, mrpc

For Consistency Model

```python
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=True python src/training.py -dataset qqp-new -add_ds paws mrpc-new -model roberta -model_type dual -lr 2e-5 -additional_cls -div kl -seed 42 -augment_reverse -tbs 12 -n_gpus 1 -s_off -maxe 2
```

For Standard Single Model

```python
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=True python src/training.py -dataset qqp-new -add_ds paws mrpc-new -model roberta -model_type single -lr 4e-5 -additional_cls -div kl -seed 42 -augment_reverse -tbs 12 -n_gpus 1 -s_off -maxe 4
```

Please check that the models should be saved in the folder `Models`. Pick specific model directory inside the `Models` directory for checkpoint finetuning. Let's call that `Models/ckptdir`

## Fine-Tuning on datasets

The models can be fine-tuned on any of the following [datasets](#dataset) .
1. paws
2. mrpc-new
3. sst2-eq: Train Size: 69393; Validation size: 872; Test Size: 5956
4. sst2-new: Train Size: 60615; Validation size: 872; Test Size: 6734
5. rte-eq
6. qnli-eq

**Note**: The original `sst2-eq` data files got corrupted and had to be regenerated. We provide two new versions `sst2-eq` and `sst2-new` as replacements. The final performance of the model, should, ideally, remain unchanged.

Replace any of the `<dataset>` mentioned above in the command below:

```python 
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=True python src/finetuning.py -ckpt <Models/ckptdir> -lr 2e-5 -dataset <dataset> -tbs 12 -maxe 3 -n_gpus 1
```

No additional fine-tuning is required for `qqp-new` dataset. Classification can be performed directly.

## Classification

After fine-tuning the following folders will get created `Models/ckptdir/finetune`

For `sst2-eq`, `rte-eq`, `qnli-eq`
```python
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=True python src/classification.py -dataset <dataset> -ebs 256 -ckpt <Models/ckptdir/finetune> -n_gpus 1
```

For `mrpc-new`, `paws`
```python
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=True python src/classification.py -dataset <dataset> -ebs 256 -ckpt <Models/ckptdir/finetune> -econs -n_gpus 1
```

For `qqp-new` (Since no fine-tuning was done for qqp-new, see that the checkpoint name points to `Models/ckptdir`)
```python
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=True python src/classification.py -dataset <dataset> -ebs 256 -ckpt <Models/ckptdir> -econs -n_gpus 1
```

The final prediction scores will be available in the `<Models/ckptdir/finetune/<dataset>>` file.

Please see additional arguments in `src/args.py` for experimentation. Evaluation on reverse candidates can be permitted through the `-erev` argument for symmetric classification datasets.

## Evaluation

### Reference Generation

First generate the reference files for each of the dataset and model for comparison using the following command:

```python
python src/create_references.py -ckpt <Models/ckptdir> -dataset <dataset> -ebs 512
```

For generation of all reference files, please use:

```python
python create_reference_json.py -modeldir Models
```
This will generate a file called `precommands.json` in the main directory. Subsequently use the following command to generate all relevant references.

```python
python create_all_references.py -i precommands.json
```

### Final Evaluation 

For final evaluation run:

```python
python src/evaluation.py -pretrain_path `Models`
```

The final evaluation results can be accessed via `FinalResults.csv` generated.

## Citation

Please cite the following paper if you find this work relevant to your application

```bibtex
@inproceedings{kumar-joshi-2022-striking,
    title = "Striking a Balance: Alleviating Inconsistency in Pre-trained Models for Symmetric Classification Tasks",
    author = "Kumar, Ashutosh  and
      Joshi, Aditya",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.148",
    pages = "1887--1895",
}
```

For any clarification, comments, or suggestions please create an issue or contact [ashutosh@iisc.ac.in](http://ashutoshml.github.io)
