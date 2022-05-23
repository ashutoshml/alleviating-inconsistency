from models import SingleModel
import torch
import torch.nn as nn
import constants as C
import pdb as pdb


class DualModel(nn.Module):
    def __init__(self, args):
        super(DualModel, self).__init__()
        self.config = args

        if self.config.model in C.AVAILABLE_MODELS:
            self.model = SingleModel(args)
        else:
            raise NotImplementedError

    def update_classifier(self, n_classes):
        self.model.update_classifier(n_classes)

    def forward(self, lr_input_ids, lr_attention_mask, rl_input_ids, rl_attention_mask, **kwargs):
        logits_0, hidden_0, logits_para_0 = self.model(lr_input_ids, lr_attention_mask, lr_token_type_ids=kwargs['lr_token_type_ids'] if 'lr_token_type_ids' in kwargs else None)
        logits_1, hidden_1, logits_para_1 = self.model(rl_input_ids, rl_attention_mask, lr_token_type_ids=kwargs['rl_token_type_ids'] if 'rl_token_type_ids' in kwargs else None) # work-around

        logits = torch.cat([logits_0.unsqueeze(0), logits_1.unsqueeze(0)])
        logits_para = None
        if self.config.additional_cls:
            logits_para = torch.cat([logits_para_0.unsqueeze(0), logits_para_1.unsqueeze(0)])

        hidden_last = None
        if hidden_0 is not None:
            hidden_last = torch.cat([hidden_0.unsqueeze(0), hidden_1.unsqueeze(0)])

        if self.config.model_type == 'dual' and not self.config.consistency:
            # pdb.set_trace()
            all_hidden = torch.cat([hidden_0, hidden_1, torch.abs(hidden_0 - hidden_1)], dim=1)
            logits_2 = self.siameseclass(all_hidden)
            logits = torch.cat([logits_0.unsqueeze(0), logits_1.unsqueeze(0), logits_2.unsqueeze(0)])
            # pdb.set_trace()

            if self.config.additional_cls:
                logits_para_2 = self.siameseclass_para(all_hidden)
                logits_para = torch.cat([logits_para_0.unsqueeze(0), logits_para_1.unsqueeze(0), logits_para_2.unsqueeze(0)])

        return logits, hidden_last, logits_para
