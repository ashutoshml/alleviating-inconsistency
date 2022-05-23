import os
import functools
import operator
import sys
import tqdm as tqdm
import pandas as pd
from args import Args
import constants as C
from pytorch_lightning import seed_everything, Trainer

import pdb as pdb

from datamodule import DataModule
from transformers import AutoTokenizer


def create_output_dicts(batch, dataset):
    if dataset.split('-')[0] == 'stsb':
        results = {'score': [b.item() for b in batch['label']]}
    else:
        results = {'prediction': [int(b.item()) for b in batch['label']],
                'confidence': [1.0 for _ in batch['label']]}

    return results


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(C.CONFIG['roberta'])
    datamodule = DataModule(args.dataset,
                            args.train_batch_size,
                            args.eval_batch_size,
                            two_side=False,
                            augment_reverse=False,
                            additional_cls=False,
                            tokenizer=tokenizer,
                            max_len=args.max_len,
                            eval_reverse=args.eval_reverse)
    datamodule.prepare_data()
    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()

    outputs = []
    for i, batch in enumerate(test_dataloader):
        outputs.append(create_output_dicts(batch, args.dataset))
        print("Inserting {}".format(i), end='\r')

    if args.dataset.split('-')[0] != 'stsb':
        prediction = functools.reduce(operator.iconcat, [x["prediction"] for x in outputs], [])
        confidence = functools.reduce(operator.iconcat, [x["confidence"] for x in outputs], [])
        results = {'prediction': prediction, 'confidence': confidence}
    else:
        results = functools.reduce(operator.iconcat, [x["score"] for x in outputs], [])
        results = {'score': results}
    outputs = results

    results_file = os.path.join(args.modeldir, 'references.csv')

    print('Writing answers to file: {}'.format(results_file))
    df = pd.DataFrame(outputs)
    df.to_csv(results_file, sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    args = Args()

    parser = args.get_generic_args()
    parser = DataModule.add_data_specific_args(args)
    parser = args.get_classification_args()

    args = parser.parse_args()

    args.n_classes = C.NCLASS_DATASET[args.dataset.split('-')[0]]
    if '.ckpt' in args.checkpoint:
        pretrain_dir = os.path.split(args.checkpoint)[0]
    else:
        pretrain_dir = args.checkpoint

    # SET SEEDS
    seed_everything(args.seed)

    if args.dataset in ['qqp-new', 'paws', 'mrpc-new']:
        args.pretrain_dir = pretrain_dir
        args.modeldir = os.path.join(args.pretrain_dir, 'finetune', args.dataset, 'classification')
    else:
        args.pretrain_dir = os.path.join(pretrain_dir, 'finetune', args.dataset)
        args.modeldir = os.path.join(args.pretrain_dir, 'classification')

    args.logdir = os.path.join(*(args.modeldir.split(os.path.sep)[1:]))
    args.logdir = os.path.join(C.LOGPATH, args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    main(args)
