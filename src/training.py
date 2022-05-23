import os
import pdb as pdb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

from datamodule import DataModule
from model import ClassificationModel
from args import Args
import constants as C


def construct_trainer_dict(args):
    trainer_dict = {}
    trainer_dict["max_epochs"] = args.max_epochs
    trainer_dict["min_epochs"] = args.min_epochs
    trainer_dict["accelerator"] = args.accelerator
    trainer_dict["precision"] = args.precision
    trainer_dict["check_val_every_n_epoch"] = args.check_val_every_n_epoch
    trainer_dict["gradient_clip_val"] = args.gradient_clip_val
    trainer_dict["overfit_batches"] = args.overfit_batches
    trainer_dict["limit_train_batches"] = args.limit_train_batches
    trainer_dict["log_every_n_steps"] = args.log_every_n_steps
    trainer_dict["accumulate_grad_batches"] = args.accumulate_grad_batches
    trainer_dict["stochastic_weight_avg"] = True

    trainer_dict["deterministic"] = True

    if args.n_gpus > 0:
        trainer_dict["gpus"] = args.n_gpus
        trainer_dict["accelerator"] = 'ddp'
        if args.n_gpus > 1:
            trainer_dict["plugins"] = DDPPlugin(find_unused_parameters=False)
    if args.tpu_cores > 0:
        trainer_dict["tpu_cores"] = args.tpu_cores
        del trainer_dict["gpus"]

    if args.profiler:
        trainer_dict['profiler'] = 'simple'

    return trainer_dict


def train(args):
    logger = WandbLogger(name=args.postfix, project="consistent-paraphrase", log_model="all", entity=C.WANDBENTITY)

    # Generator
    if args.checkpoint is None:
        ckpt_callback = ModelCheckpoint(dirpath=args.modeldir,
                                        filename='sample-{epoch:02d}-{val_loss_epoch:.2f}',
                                        monitor=args.monitor,
                                        save_top_k=args.save_top_k)
        es_callback = EarlyStopping(monitor=args.monitor, min_delta=0.0, patience=args.patience, verbose=True, mode=args.metric_mode)
        classifier = ClassificationModel(args)

    else:
        ckpt_callback = ModelCheckpoint(dirpath=args.modeldir)
        es_callback = EarlyStopping(monitor=args.monitor, min_delta=0.0, patience=args.patience, verbose=True, mode=args.metric_mode)
        classifier = ClassificationModel.load_from_checkpoint(args.checkpoint)

    two_side = True if args.model_type == 'dual' else False

    datamodule = DataModule(args.dataset,
                            args.train_batch_size,
                            args.eval_batch_size,
                            two_side=two_side,
                            augment_reverse=args.augment_reverse,
                            additional_cls=args.additional_cls,
                            tokenizer=classifier.tokenizer,
                            max_len=args.max_len,
                            append_multiple_train=False if args.additional_dataset is None or args.additional_dataset == '' else True)
    datamodule.prepare_data()
    datamodule.setup('fit', args.additional_dataset)

    classifier.set_total_steps(len(datamodule.train_dataloader().dataset))

    logger.watch(classifier)

    trainer_dict = construct_trainer_dict(args)
    trainer = Trainer(**trainer_dict, callbacks=[ckpt_callback, es_callback])

    trainer.fit(classifier, datamodule)


def construct_postfix(args):
    args.postfix = 'dataset={}'.format(args.dataset)
    args.postfix += '-model={}'.format(args.model)
    args.postfix += '-consistency={}'.format(args.consistency)
    args.postfix += '-seed={}'.format(args.seed)
    args.postfix += '-additional_cls={}'.format(args.additional_cls)
    args.postfix += '-augment_reverse={}'.format(args.augment_reverse)
    args.postfix += '-overfit_batches={}'.format(args.overfit_batches)

    if args.consistency:
        args.postfix += '-divergence={}'.format(args.divergence)
    if args.additional_dataset is not None:
        args.postfix += '-add_ds={}'.format(''.join(args.additional_dataset))
    if hasattr(args, 'scheduler_off'):
        args.postfix += '-sch_off={}'.format(args.scheduler_off)

    args.postfix += '-train'


if __name__ == '__main__':
    args = Args()

    parser = args.get_generic_args()
    parser = args.get_trainer_args()
    parser = args.get_optimizer_args()
    parser = args.get_callback_args()

    parser = DataModule.add_data_specific_args(args)
    parser = ClassificationModel.add_model_args(args)

    args = parser.parse_args()
    args.n_classes = C.NCLASS_DATASET[args.dataset.split('-')[0]]

    # SET SEEDS
    seed_everything(args.seed)
    construct_postfix(args)

    args.modeldir = os.path.join(C.MODELPATH, args.postfix)

    os.makedirs(args.modeldir, exist_ok=True)

    train(args)
