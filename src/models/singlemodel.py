import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
import constants as C
from collections import OrderedDict
import os

import pdb as pdb


class SingleModel(nn.Module):
    def __init__(self, args):
        super(SingleModel, self).__init__()
        self.config = args

        model_config = AutoConfig.from_pretrained(C.CONFIG[self.config.model])
        self.model = AutoModel.from_pretrained(C.CONFIG[self.config.model], model_config)

        if self.config.additional_cls:
            self.classifier_para = self.add_classifier(2)
            self.classifier = self.add_classifier(self.config.n_classes)
            print('Adding classifier ')
        else:
            self.classifier = self.add_classifier(self.config.n_classes)
            print('Adding classifier ')

    def add_classifier(self, n_classes):
        classifier = nn.Sequential(nn.Linear(C.HIDDEN_DIM[self.config.model]['normal'], C.HIDDEN_DIM[self.config.model]['normal']),
                                   nn.Tanh(),
                                   nn.Dropout(self.config.dropout_prob),
                                   nn.Linear(C.HIDDEN_DIM[self.config.model]['normal'], n_classes))
        for x in classifier:
            self._init_weights(x)
        return classifier

    def update_classifier(self, n_classes):
        self.classifier = self.add_classifier(n_classes)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        input_ids =  input_ids
        attention_mask = attention_mask
        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=kwargs['lr_token_type_ids'] if 'lr_token_type_ids' in kwargs else None)

        last_hidden = outputs[0]
        first_token = last_hidden[:, 0]

        # logits = self.classifier(pooled)
        logits = self.classifier(first_token)
        # logits = self.classifier_ran(outputs[1])

        logits_para = None
        if self.config.additional_cls:
            second_token = last_hidden[:, 1]
            logits_para = self.classifier_para(second_token)
            # logits_para = self.classifier_para(pooled_para)

        hidden = None
        if self.config.model_type == 'dual' and not self.config.consistency:
            hidden = first_token

        return logits, hidden, logits_para
