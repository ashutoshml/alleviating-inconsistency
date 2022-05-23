import os
import sys
import pdb as pdb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule import DataModule
from model import ClassificationModel
from args import Args
import constants as C


def construct_trainer_dict(args):
    trainer_dict = {}
    trainer_dict['max_epochs'] = args.max_epochs
    trainer_dict['min_epochs'] = args.min_epochs
    if args.n_gpus > 0:
        trainer_dict["gpus"] = args.n_gpus
        trainer_dict["accelerator"] = 'ddp'
    if args.tpu_cores > 0:
        trainer_dict["tpu_cores"] = args.tpu_cores
        del trainer_dict["gpus"]
    return trainer_dict

def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])

def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])

def get_best_checkpoint(pretrain_dir, monitor="val_loss_epoch"):
    ckpt_files = [x[:-5] for x in os.listdir(pretrain_dir) if ".ckpt" in x]
    prefix_name = ckpt_files[0].split("-")[0]
    ckpt_files = [x.replace("{}-".format(prefix_name), "") for x in ckpt_files]

    check_mon = [a.split(monitor)[-1].replace("=", "") for a in ckpt_files]
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

    ckpt_files_subset = [
        x for x in ckpt_files if "{}={}".format(monitor, check_mon[check_mon_idx]) in x
    ]

    check_epochs = [a.split("-")[0].replace("epoch=", "") for a in ckpt_files_subset]
    check_epochs_idx = argmax([int(x) for x in check_epochs])

    selected_ckpt = [
        x
        for x in ckpt_files_subset
        if "epoch={}".format(check_epochs[check_epochs_idx]) in x
    ][0]
    best_ckpt = os.path.join(
        pretrain_dir, "{}-{}.ckpt".format(prefix_name, selected_ckpt)
    )

    return best_ckpt


def finetune(args):
    logger = WandbLogger(project="consistent-paraphrase", log_model="all", entity=None)

    # Generator
    if args.checkpoint is None:
        print('No checkpoint available for finetuning. Exit setup')
        sys.exit()
    else:
        ckpt_callback = ModelCheckpoint(dirpath=args.modeldir,
                                        filename='finetune-{epoch:02d}-{val_loss_epoch:.2f}',
                                        monitor=args.monitor,
                                        save_top_k=1)

        if '.ckpt' in args.checkpoint:
            classifier = ClassificationModel.load_from_checkpoint(args.checkpoint)
            print('Loading from the checkpoint provided')
        else:
            best_ckpt = get_best_checkpoint(args.pretrain_dir)
            print('Loading from the best available checkpoint in {}: {}'.format(args.pretrain_dir, best_ckpt))
            classifier = ClassificationModel.load_from_checkpoint(best_ckpt)

        es_callback = EarlyStopping(monitor=args.monitor, min_delta=0.0, patience=args.patience, verbose=True, mode=args.metric_mode)
        classifier.config.n_classes = args.n_classes
        classifier.update_is_finetune()

    two_side = True if classifier.config.model_type == 'dual' else False
    additional_cls = classifier.config.additional_cls
    classifier.learning_rate = args.learning_rate

    datamodule = DataModule(args.dataset,
                            args.train_batch_size,
                            args.eval_batch_size,
                            two_side=two_side,
                            augment_reverse=args.augment_reverse,
                            additional_cls=additional_cls,
                            tokenizer=classifier.tokenizer,
                            max_len=args.max_len)
    datamodule.prepare_data()
    datamodule.setup('fit')

    trainer_dict = construct_trainer_dict(args)
    trainer = Trainer(**trainer_dict, logger=logger, callbacks=[ckpt_callback, es_callback])
    trainer.fit(classifier, datamodule)


if __name__ == '__main__':
    args = Args()

    parser = args.get_generic_args()
    parser = args.get_callback_args()
    parser = args.get_optimizer_args()
    parser = args.get_finetuning_args()
    parser = DataModule.add_data_specific_args(args)

    args = parser.parse_args()

    args.n_classes = C.NCLASS_DATASET[args.dataset.split('-')[0]]
    if '.ckpt' in args.checkpoint:
        pretrain_dir = os.path.split(args.checkpoint)[0]
    else:
        pretrain_dir = args.checkpoint

    # SET SEEDS
    seed_everything(args.seed)

    args.modeldir = os.path.join(pretrain_dir, 'finetune', '{}'.format(args.dataset))
    args.logdir = os.path.join(*(args.modeldir.split(os.path.sep)[1:]))
    args.logdir = os.path.join(C.LOGPATH, args.logdir)
    args.pretrain_dir = pretrain_dir
    
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    finetune(args)
