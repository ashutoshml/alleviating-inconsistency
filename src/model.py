import functools
import operator
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# import torch_xla.core.xla_model as xm

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import pdb as pdb
import constants as C

from models.singlemodel import SingleModel
from models.dualmodel import DualModel
from utils.lambda_annealing import WeightAnnealing, LossDivergence

from torch.optim import Adam, Adadelta, SGD
from prettytable import PrettyTable


class ClassificationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.config = args
        self.learning_rate = self.config.learning_rate
        self.is_finetune = False
        self.is_classify = False
        self.is_classify_cons = False
        self.total_steps = 1000

        try:
            if self.config.model_type == 'single':
                self.model = SingleModel(self.config)
            elif self.config.model_type == 'debugc':
                model_config = AutoConfig.from_pretrained(C.CONFIG[self.config.model])
                self.model = AutoModelForSequenceClassification.from_pretrained(C.CONFIG[self.config.model], config=model_config)
            else:
                self.model = DualModel(self.config)
            self.tokenizer = AutoTokenizer.from_pretrained(C.CONFIG[self.config.model])

            if self.config.additional_cls:
                self._resize_token_emb()
        except Exception as e:
            print(e)
            pdb.set_trace()
            raise NotImplementedError


        if self.config.model_type == 'dual' and self.config.consistency:
            self.kl_weight_annealing = WeightAnnealing(self.config.c_lambda,
                                                       self.config.warmup_lambda,
                                                       self.config.lambda_update_steps,
                                                       self.config.maxclambda,
                                                       self.config.n_gpus if self.config.n_gpus > 1 else 1,
                                                       self.config.train_batch_size if hasattr(self.config, 'train_batch_size') else 1)

        self.train_ep = 0
        self.count_parameters()

    def count_parameters(self):
        model = self.model
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def update_is_finetune(self):
        self.is_finetune = True
        self.model.update_classifier(self.config.n_classes)

    def update_is_classify(self):
        self.is_finetune = False
        self.is_classify = True

    def update_is_classify_cons(self):
        self.is_finetune = False
        self.is_classify = False
        self.is_classify_cons = True

    def set_total_steps(self, train_size=1000):
        self.total_steps = (
                (train_size // (self.config.train_batch_size * max(1, self.config.n_gpus)))
                // self.config.accumulate_grad_batches
                * float(self.config.max_epochs)
            )

    def _resize_token_emb(self):
        special_tokens_dict = {'additional_special_tokens': C.SPECIAL_TOKENS}
        _ = self.tokenizer.add_special_tokens(special_tokens_dict)
        try:
            self.model.model.resize_token_embeddings(len(self.tokenizer))
        except:
            self.model.model.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch, val=False):
        if self.config.model_type == 'debugc':
            outputs = self.model(input_ids=batch['lr_ids'],
                                 attention_mask=batch['lr_mask'],
                                 token_type_ids=batch['lr_token_type_ids'] if 'lr_token_type_ids' in batch else None,
                                 labels=batch['label'] if 'label' in batch else None)
            try:
                loss = outputs.loss
            except:
                loss = None
            # pdb.set_trace()
            return [outputs.logits], loss
        else:
            outputs = self.model(batch['lr_ids'],
                                 batch['lr_mask'],
                                 lr_token_type_ids=batch['lr_token_type_ids'] if 'lr_token_type_ids' in batch else None,
                                 rl_input_ids=batch['rl_ids'] if 'rl_ids' in batch else None,
                                 rl_attention_mask=batch['rl_mask'] if 'rl_mask' in batch else None,
                                 rl_token_type_ids=batch['rl_token_type_ids'] if 'rl_token_type_ids' in batch else None)
            if 'label' in batch and batch['label'] is not None:
                loss = self._calculate_loss(outputs, batch['label'], val)
                return outputs, loss

            return outputs, None

    def classify(self, batch):
        outputs, _ = self(batch, True)
        relevant_logits = self.get_relevant_logits(outputs)
        outputs = self.get_argmax_outputs(relevant_logits)
        return outputs

    def regress(self, batch):
        outputs, _ = self(batch, True)
        logits = self.get_relevant_logits(outputs)
        outputs = {
            'score': logits.squeeze()
        }
        return outputs

    def _loss_single(self, outputs, label):
        loss_fct = nn.CrossEntropyLoss()
        loss_mse_fct = nn.MSELoss()

        if self.config.n_classes == 1 and (self.is_finetune or self.is_classify):
            loss = loss_mse_fct(outputs[0].squeeze(), label)
        else:
            if self.is_finetune or self.is_classify:
                loss = loss_fct(outputs[0], label)
            else:
                if self.config.additional_cls:
                    loss = loss_fct(outputs[2], label)
                else:
                    loss = loss_fct(outputs[0], label)
        # pdb.set_trace()
        self.log('ce_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def _loss_dual_consistency(self, outputs, label, val=False):
        loss_fct = nn.CrossEntropyLoss()
        loss_mse_fct = nn.MSELoss()
        loss_kl_fct = LossDivergence(self.kl_weight_annealing, self.config.divergence)

        if self.config.n_classes == 1 and (self.is_finetune or self.is_classify):
            loss = loss_mse_fct(outputs[0][1].squeeze(), label)
        else:
            if self.is_finetune or self.is_classify:
                loss = loss_fct(outputs[0][1], label)
            else:
                if self.config.additional_cls:
                    dim = 2
                else:
                    dim = 0

                loss = loss_fct(outputs[dim][0], label)
                loss += loss_fct(outputs[dim][1], label)
                divloss = self.kl_weight_annealing.scheduler_lambda(val)*loss_kl_fct.get_loss(outputs[dim][0], outputs[dim][1])
                self.log('ce_loss', loss, prog_bar=True, sync_dist=True)
                c_lam = self.kl_weight_annealing.get_lam()
                self.log('l_step', self.kl_weight_annealing.step, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
                self.log('c_lam', c_lam, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
                self.log('divloss', divloss, prog_bar=True, sync_dist=True)
                loss += divloss
        return loss


    def get_relevant_logits(self, outputs):
        relevant_logits = None

        if self.config.model_type == 'single':
            relevant_logits = outputs[0]
            if not self.is_classify and not self.is_finetune:
                if self.config.additional_cls:
                    relevant_logits = outputs[2]
        elif self.config.model_type == 'debugc':
            relevant_logits = outputs[0]
        else:
            if self.config.consistency:
                relevant_logits = outputs[0][0]
                if not self.is_classify and not self.is_finetune:
                    if self.config.additional_cls:
                        relevant_logits = outputs[2][0]
            else:
                relevant_logits = outputs[0][2]
                if not self.is_classify and not self.is_finetune:
                    if self.config.additional_cls:
                        relevant_logits = outputs[2][2]
        return relevant_logits

    def get_argmax_outputs(self, relevant_logits):
        pred_idx = torch.argmax(relevant_logits, dim=1, keepdim=True)
        labels_hat = pred_idx.squeeze(-1)
        softlogits = F.softmax(relevant_logits, dim=1)
        confs = torch.gather(softlogits, 1, pred_idx).squeeze(-1)

        outputs = {
                'prediction': labels_hat,
                'confidence': confs
        }
        return outputs

    def get_accuracy(self, labels_hat, labels):
        val_acc = torch.sum(labels == labels_hat).item() / (len(labels) * 1.0)
        val_acc = torch.tensor(val_acc)
        return val_acc

    def _calculate_loss(self, outputs, label, val=False):
        if self.config.model_type == 'single':
            loss = self._loss_single(outputs, label)
        elif self.config.model_type == 'dual':
            loss = self._loss_dual_consistency(outputs, label, val)
        return loss

    def training_step(self, batch, batch_idx):
        _, loss = self(batch, False)
        if hasattr(self.config, 'scheduler_off') and self.config.scheduler_off:
            pass
        else:
            self.log('lr', self.lr_scheduler.get_last_lr()[-1], prog_bar=True, sync_dist=True)
        return loss

    def on_epoch_end(self):
        self.train_ep += 1

    def validation_step(self, batch, batch_idx):
        outputs, loss = self(batch, val=True)

        relevant_logits = self.get_relevant_logits(outputs)
        outputs = self.get_argmax_outputs(relevant_logits)
        labels_hat = outputs['prediction']

        if self.config.n_classes != 1:
            val_acc = self.get_accuracy(labels_hat, batch['label'])
            self.log('val_acc', val_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            val_acc = 0.

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_loss': loss, 'val_acc': val_acc}

    def test_step(self, batch, batch_idx):
        if self.is_classify or self.is_classify_cons:
            if self.config.n_classes != 1:
                result = self.classify(batch)
            else:
                result = self.regress(batch)
        else:
            raise NotImplementedError
        return result

    def test_epoch_end(self, outputs):
        if self.config.n_classes != 1:
            prediction = functools.reduce(operator.iconcat, [x["prediction"] for x in outputs], [])
            confidence = functools.reduce(operator.iconcat, [x["confidence"] for x in outputs], [])
            results = {
                'prediction': [int(x.item()) for x in prediction],
                'confidence': [x.item() for x in confidence],
            }
        else:
            results = functools.reduce(operator.iconcat, [x["score"] for x in outputs], [])
            results = [x.item() for x in results]
            results = {
                    'score': results,
            }
        self.test_results = results
        return results

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        # no_decay = []
        model = self.model
        optimizer_grouped_params = [
                {
                    'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.weight_decay
                },
                {
                    'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
                ]
        optimizer = AdamW(optimizer_grouped_params, lr=self.learning_rate, eps=self.config.epsilon)

        if hasattr(self.config, 'scheduler_off') and self.config.scheduler_off:
            return [optimizer]
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.total_steps)
            self.lr_scheduler = scheduler
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }

            return [optimizer], [scheduler]

    def optimizer_step(self, epoch=None,
                       batch_idx=None, optimizer=None,
                       optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None,
                       using_lbfgs=None):

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        if hasattr(self.config, 'scheduler_off') and self.config.scheduler_off:
            pass
        else:
            self.lr_scheduler.step()

    @classmethod
    def add_model_args(cls, args):
        return args.get_model_args()
