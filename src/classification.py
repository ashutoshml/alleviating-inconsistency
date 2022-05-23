import os
import sys
import functools
import operator
import pdb as pdb
from tqdm import tqdm

from args import Args
from model import ClassificationModel
from datamodule import DataModule

import constants as C
import pandas as pd

from pytorch_lightning import seed_everything, Trainer


def construct_trainer_dict(args):
    trainer_dict = {}
    if args.n_gpus > 0:
        trainer_dict["gpus"] = args.n_gpus
    if args.tpu_cores > 0:
        trainer_dict["tpu_cores"] = args.tpu_cores
        del trainer_dict["gpus"]
    return trainer_dict

def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])

def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])

def get_best_checkpoint(pretrain_dir, monitor='val_loss_epoch'):
    ckpt_files = [x[:-5] for x in os.listdir(pretrain_dir) if '.ckpt' in x]
    prefix_name = ckpt_files[0].split('-')[0]
    ckpt_files = [x.replace('{}-'.format(prefix_name), '') for x in ckpt_files]

    check_mon = [a.split(monitor)[-1].replace('=', '') for a in ckpt_files]
    if monitor == "val_loss_epoch":
        check_mons = []
        for x in check_mon:
            try:
                check_mons.append(float(x))
            except:
                try:
                    check_mons.append(float(x.split('-')[0]))
                except:
                    continue
        check_mon_idx = argmin(check_mons)
    elif monitor == "val_acc_epoch":
        check_mons = []
        for x in check_mon:
            try:
                check_mons.append(float(x))
            except:
                try:
                    check_mons.append(float(x.split('-')[0]))
                except:
                    continue
        check_mon_idx = argmax(check_mons)
    else:
        raise NotImplementedError

    ckpt_files_subset = [x for x in ckpt_files if '{}={}'.format(monitor, check_mon[check_mon_idx]) in x]

    check_epochs = [a.split('-')[0].replace('epoch=', '') for a in ckpt_files_subset]
    check_epochs_idx = argmax([int(x) for x in check_epochs])

    selected_ckpt = [x for x in ckpt_files_subset if 'epoch={}'.format(check_epochs[check_epochs_idx]) in x][0]
    best_ckpt = os.path.join(pretrain_dir, '{}-{}.ckpt'.format(prefix_name, selected_ckpt))
    # import pdb; pdb.set_trace()

    return best_ckpt


def classify(args):
    try:
        best_ckpt = get_best_checkpoint(args.pretrain_dir)
    except:
        print('No checkpoints found in {} \t Stopping Classification'.format(args.pretrain_dir))
        sys.exit()
    classifier = ClassificationModel.load_from_checkpoint(best_ckpt)
    print('Loading from the best available checkpoint in {}: {}'.format(args.pretrain_dir, best_ckpt))

    two_side = True if classifier.config.model_type == 'dual' else False
    additional_cls = classifier.config.additional_cls

    # DUAL WITHOUT CONSISTENCY IS SIAMESE
    datamodule = DataModule(args.dataset,
                            args.train_batch_size,
                            args.eval_batch_size,
                            two_side=two_side,
                            augment_reverse=False,
                            additional_cls=additional_cls,
                            tokenizer=classifier.tokenizer,
                            max_len=args.max_len,
                            eval_reverse=args.eval_reverse)
    datamodule.prepare_data()
    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()

    if args.eval_consistency:
        classifier.update_is_classify_cons()
    else:
        classifier.update_is_classify()
    classifier.eval()
    classifier.freeze()

    trainer_dict = construct_trainer_dict(args)
    trainer = Trainer(**trainer_dict)
    trainer.test(classifier, test_dataloaders=test_dataloader)
    outputs = classifier.test_results

    if args.eval_consistency:
        results_file = os.path.join(args.modeldir, 'results-cons-{}-rev-{}.csv'.format(args.eval_consistency, args.eval_reverse))
    else:
        results_file = os.path.join(args.modeldir, 'results-rev-{}.csv'.format(args.eval_reverse))

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

    classify(args)
