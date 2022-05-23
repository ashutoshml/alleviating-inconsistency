import argparse
import constants as C
from attrdict import AttrDict


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Argument Initializer')

    def get_generic_args(self):
        generic_args = self.parser.add_argument_group('generic_args')

        generic_args.add_argument('-seed', '--seed', type=int, default=42, help='Seed to be set')
        generic_args.add_argument('-n_gpus', '--n_gpus', type=int, default=0, help='Number of gpus to use')
        generic_args.add_argument('-tpu_cores', '--tpu_cores', type=int, default=0, help='TPU Cores to use')

        return self.parser

    def get_callback_args(self):
        callback_args = self.parser.add_argument_group('callback_args')

        callback_args.add_argument('-monitor', '--monitor', default='val_loss', choices=['val_loss', 'val_acc'], help='What metric to monitor')
        callback_args.add_argument('-patience', '--patience', type=int, default=3, help='Number of epochs till no improvement for training to stop')
        callback_args.add_argument('-save_top_k', '--save_top_k', type=int, default=3, help='Best k models according to monitor')
        callback_args.add_argument('-metric_mode', '--metric_mode', default='min', choices=['min', 'max', 'auto'], help='Metric mode for model saving')

        return self.parser

    def get_data_module_args(self):
        data_module_args = self.parser.add_argument_group('data_module_args')

        data_module_args.add_argument('-dataset', '--dataset', default='qqp', help='Dataset for training and evaluation')
        data_module_args.add_argument('-add_ds', '--additional_dataset', default=None, nargs='+', help='Additional Dataset to be used?')
        data_module_args.add_argument('-tbs', '--train_batch_size', type=int, default=12, help='Batch size for training')
        data_module_args.add_argument('-ebs', '--eval_batch_size', type=int, default=12, help='Batch size for validation')
        data_module_args.add_argument('-max_len', '--max_len', type=int, default=256, help='Max length of text to be considered')
        data_module_args.add_argument('-augment_reverse', '--augment_reverse', action='store_true', help='Augment reverse of dataset')
        data_module_args.add_argument('-additional_cls', '--additional_cls', action='store_true', help='Add additional CLS token for paraphrase classification')
        data_module_args.add_argument('-erev', '--eval_reverse', action='store_true', help='Evaluation for reverse sentence1 <-> sentence2')

        return self.parser

    def get_optimizer_args(self):
        optimizer_args = self.parser.add_argument_group('optimizer_args')

        optimizer_args.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Weight decay term during optimization of loss')
        optimizer_args.add_argument('-lr', '--learning_rate', type=float, default=4e-5, help='Learning rate for the optimizer')
        optimizer_args.add_argument('-eps', '--epsilon', type=float, default=1e-8, help='Max allowable epsilon')
        optimizer_args.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Warm up steps for lr scheduler')
        optimizer_args.add_argument('-s_off', '--scheduler_off', action='store_true', help='Scheduler to work?')

        return self.parser

    def get_finetuning_args(self):
        finetuning_args = self.parser.add_argument_group('finetuning_args')
        # get checkpoint to load trained-model
        finetuning_args.add_argument('-ckpt', '--checkpoint', type=str, required=True, help='Provide checkpoint from which to start fine-tuning the model')
        finetuning_args.add_argument('-maxe', '--max_epochs', type=int, default=5, help='Number of epochs for training')
        finetuning_args.add_argument('-mine', '--min_epochs', type=int, default=3, help='Minimum Number of epochs for training')

        return self.parser

    def get_trainer_args(self):
        trainer_args = self.parser.add_argument_group('trainer_args')

        trainer_args.add_argument('-gc', '--gradient_clip_val', type=float, default=0, help='Gradient clipping value')
        trainer_args.add_argument('-dpout', '--dropout_prob', type=float, default=0.1, help='Drop out probability')
        trainer_args.add_argument('-maxe', '--max_epochs', type=int, default=10, help='Number of epochs for training')
        trainer_args.add_argument('-mine', '--min_epochs', type=int, default=5, help='Minimum Number of epochs for training')
        trainer_args.add_argument('-acc', '--accelerator', default=None, choices=['dp', 'ddp', 'ddp_cpu', 'ddp2'], help='Accelerator to use')
        trainer_args.add_argument('-ofit', '--overfit_batches', type=float, default=0.0, help='Overfit a percent of training batches')
        trainer_args.add_argument('-prec', '--precision', type=int, default=32, choices=[16,32], help='Precision')
        trainer_args.add_argument('-check_val_ep', '--check_val_every_n_epoch', type=int, default=1, help='Check val every n epoch')
        trainer_args.add_argument('-acc_grad_batch', '--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation batches')
        trainer_args.add_argument('-log_every_n_steps', '--log_every_n_steps', type=int, default=100, help='Log in every n steps')
        trainer_args.add_argument('-limit_train_batches', '--limit_train_batches', type=float, default=1., help='Fraction of training set to be used')
        trainer_args.add_argument('-profiler', '--profiler', action='store_true', help='Enable profiler')

        retraining_args = self.parser.add_argument_group('retraining_args')

        retraining_args.add_argument('-ckpt', '--checkpoint', default=None, help='Checkpoint to load for re-training')

        return self.parser

    def get_model_args(self):
        model_args = self.parser.add_argument_group('model_args')

        model_args.add_argument('-model', '--model', default='roberta', choices=['bert', 'roberta', 'albert'], help='Which model to use for learning?')
        model_args.add_argument('-model_type', '--model_type', required=True, choices=['single', 'dual', 'debugc'], help='Dual model or single model ?')
        model_args.add_argument('-consistency', '--consistency', action='store_false', help='Do consistency training with dual model')
        model_args.add_argument('-initializer_range', '--initializer_range', type=float, default=0.02, help='For initializing layers')
        # Specific to consistency training
        model_args.add_argument('-warmup_lambda', '--warmup_lambda', type=int, default=2000, help='Warmup steps for lambda')
        model_args.add_argument('-lus', '--lambda_update_steps', type=int, default=500, help='Update lambda after these many steps')
        model_args.add_argument('-c_lambda', '--c_lambda', type=float, default=0.0, help='Starting point for c_lambda')
        model_args.add_argument('-maxclambda', '--maxclambda', type=float, default=100.0, help='Max value for c_lambda')
        model_args.add_argument('-div', '--divergence', default='js', choices=['kl', 'js'], help='Which divergence to minimize')

        return self.parser

    def get_classification_args(self):
        classification_args = self.parser.add_argument_group('classification_args')

        classification_args.add_argument('-ckpt', '--checkpoint', required=True, help='Checkpoint for classification')
        classification_args.add_argument('-econs', '--eval_consistency', action='store_true', help='Evaluate for consistency')

        return self.parser

    def get_splitter_set_args(self):
        splitter_set_args = self.parser.add_argument_group('splitter_set_args')

        splitter_set_args.add_argument('-dataset', '--dataset', required=True, help='Dataset to use')
        splitter_set_args.add_argument('-ets', '--equal_testset', action='store_true', help='Each class in test set will have equal data points')
        splitter_set_args.add_argument('-ott', '--overwrite_traintest', action='store_true', help='Overwrite earlier created train and test files?')
        splitter_set_args.add_argument('-f', '--fraction', type=float, default=0.1, help='Fraction of points in test set - 0 to 1')

        return self.parser

    def get_evaluation_args(self):
        evaluation_args = self.parser.add_argument_group('evaluation_args')

        evaluation_args.add_argument('-dataset', '--dataset', type=str, default=None, help='Dataset to use')
        evaluation_args.add_argument('-pretrain_path', '--pretrain_path', type=str, default=None, help='If this is given then all other input file parameters are over-written')
        evaluation_args.add_argument('-ref', '--references', type=str, default=None, help='Ground Truth File')
        evaluation_args.add_argument('-pred', '--predictions', type=str, default=None, help='Predictions File')
        evaluation_args.add_argument('-save_file', '--save_file', type=str, default='FinalResults.csv', help='Save File path')

        evaluation_args.add_argument('-metric', '--metric', default=None, nargs='+', choices=[None, 'accuracy', 'f1', 'glue', 'matthews_correlation', 'pearsonr', 'spearmanr', 'mse', 'precision', 'recall'], help='Metric to be calc')

        return self.parser

