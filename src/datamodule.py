import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, IterableDataset, DataLoader

from datasets import load_dataset, concatenate_datasets, Value
import constants as C
import pdb as pdb
import pytorch_lightning as pl

from itertools import repeat


class DataModule(pl.LightningDataModule):
    tags = {
        'cola': ['sentence'],
        'sst2': ['sentence'],
        'mrpc': [('sentence1', 'sentence2')],
        'stsb': [('sentence1', 'sentence2')],
        'rte': [('sentence1', 'sentence2')],
        'wnli': [('sentence1', 'sentence2')],
        'paws': [('sentence1', 'sentence2')],
        'qqp': [('question1', 'question2')],
        'qnli': [('question', 'sentence')],
        'trec': ['text']
    }

    tokenize_labels = ['input_ids', 'attention_mask', 'token_type_ids']
    format_labels = []

    def __init__(self, dataset, train_batch_size,
                 eval_batch_size, two_side=False,
                 augment_reverse=False, additional_cls=False,
                 tokenizer=None, max_len=200,
                 append_multiple_train=False, append_multiple_test=False,
                 eval_reverse=False):
        super().__init__()
        self.dataset = dataset

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.two_side = two_side
        self.augment_reverse = augment_reverse
        self.additional_cls = additional_cls

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.append_multiple_train = append_multiple_train
        self.append_multiple_test = append_multiple_test
        self.eval_reverse = eval_reverse

    def resolve_columns(self, data, dataset):
        for colpairs in C.RENAME_COLS[dataset.split('-')[0]][self.dataset.split('-')[0]]:
            if colpairs[0] != colpairs[1]:
                data = data.rename_column(colpairs[0], colpairs[1])
            if colpairs[1] in ['idx']:
                newfeatures = data.features.copy()
                newfeatures['idx'] = Value('int64')
                data = data.cast(newfeatures)
            if colpairs[1] in ['label']:
                newfeatures = data.features.copy()
                newfeatures['label'] = Value('int64')
                data = data.cast(newfeatures)
        return data

    def add_dataset(self, dataset, data):
        print('Adding dataset: {}'.format(dataset))
        if 'train' in data and 'val' in data:
            datanew = self.get_dataset(dataset, 'fit')
            datanew = {'train': datanew[0], 'val': datanew[1]}
        else:
            datanew = self.get_dataset(dataset, 'test')
            datanew = {'test': datanew[0]}

        for key in data:
            datanew[key] = self.resolve_columns(datanew[key], dataset)
            try:
                datanew[key] = datanew[key].cast(data[key].features)
                data[key] = concatenate_datasets([data[key], datanew[key]])
            except Exception as e:
                print(e)
                pdb.set_trace()

        return data

    def get_dataset(self, dataset, stage=None):
        try:
            datadir = os.path.join(C.DATAPATH_CLASS, dataset)
        except:
            datadir = None
        if stage == 'fit' or stage is None:
            if dataset.split('-')[0] in C.GLUE_TASKS:
                try:
                    if dataset in C.GLUE_TASKS:
                        data = load_dataset('glue', dataset)
                        data = [data['train'], data['validation']]
                    else:
                        data = load_dataset('json', data_files={'train': os.path.join(datadir, 'train.json'),
                                                                'validation': os.path.join(datadir, 'validation.json')
                                                               }, split=['train', 'validation'])
                except Exception as e:
                    raise NotImplementedError('Dataset not found')
            else:
                try:
                    if len(dataset.split('-')) == 1:
                        if dataset == 'paws':
                            data = load_dataset('paws', 'labeled_final')
                        else:
                            data = load_dataset(dataset)
                        data = [data['train'], data['validation']]
                    else:
                        data = load_dataset('json', data_files={'train': os.path.join(datadir, 'train.json'),
                                                                'validation': os.path.join(datadir, 'validation.json')
                                                               }, split=['train', 'validation'])
                except Exception as e:
                    raise NotImplementedError('Dataset not found')

        if stage == 'test' or stage is None:
            if dataset.split('-')[0] in C.GLUE_TASKS:
                try:
                    if dataset in C.GLUE_TASKS:
                        data = load_dataset('glue', dataset)
                        data = [data['validation']]
                    else:
                        data = load_dataset('json', data_files={'test': os.path.join(datadir, 'test.json')
                                                               }, split=['test'])
                except Exception as e:
                    raise NotImplementedError('Dataset not found')
            else:
                try:
                    if len(dataset.split('-')) == 1:
                        if dataset == 'paws':
                            data = load_dataset('paws', 'labeled_final')
                            data = [data['test']]
                        else:
                            data = load_dataset(dataset)
                            data = [data['validation']]
                    else:
                        data = load_dataset('json', data_files={'test': os.path.join(datadir, 'test.json')
                                                               }, split=['test'])
                except Exception as e:
                    raise NotImplementedError('Dataset not found')

        return data

    def prepare_data(self):
        _ = self.get_dataset(self.dataset)

    def rename_cols(self, data, feature_label=None):
        for key in self.tokenize_labels:
            if key in data.features:
                if key == 'token_type_ids':
                    data = data.rename_column(key, '{}_token_type_ids'.format(feature_label))
                    self.format_labels.append('{}_token_type_ids'.format(feature_label))
                else:    
                    data = data.rename_column(key, '{}_{}'.format(feature_label, key.split('_')[-1]))
                    self.format_labels.append('{}_{}'.format(feature_label, key.split('_')[-1]))
        return data

    def _set_format(self, data):
        if len(self.format_labels) > 1:
            if 'label' in data.features:
                self.format_labels += ['label']
            data.set_format(type='torch', columns=self.format_labels)

        return data

    def _mapper_func_double(self, examples):
        tags = self.tags[self.dataset.split('-')[0]][0]
        assert len(tags) == 2
        s1 = []
        label = []
        s2 = []
        for x, y, l in zip(examples[tags[0]], examples[tags[1]], examples['label']):
            s1 += [x, y]
            label += [l, l]
            s2 += [y, x]

        return {tags[0]: s1, tags[1]: s2, 'label': label}

    def _mapper_func_tokens(self, examples):
        tags = self.tags[self.dataset.split('-')[0]][0]
        assert len(tags) == 2
        s1 = []
        label = []
        s2 = []
        for x, y, l in zip(examples[tags[0]], examples[tags[1]], examples['label']):
            s1 += [x, y]
            label += [l, l]
            s2 += [y, x]

        return {tags[0]: s1, tags[1]: s2, 'label': label}

    def double_data(self, data):
        print('Doubling train data because -augment_reverse was set to True')
        size = len(data)
        data = data.map(self._mapper_func_double, batched=True, remove_columns=data.column_names)
        assert len(data) == 2*size
        return data

    def add_special_token(self, data):
        print('Adding special token during data reading')
        data = data.map(self._mapper_func_tokens, batched=True, remove_columns=data.column_names)

    def appender(self, x):
        return '{} {}'.format(C.SPECIAL_TOKENS[0], x).strip()

    def _apply_tokenizer(self, data):
        prefix = ['lr', 'rl']
        for key in self.tags[self.dataset.split('-')[0]]:
            try:
                if isinstance(key, tuple):
                    if self.additional_cls:
                        try:
                            data = data.map(lambda x: self.tokenizer(list(zip(*[map(self.appender, x[k]) if k == key[0] else x[k] for k in key])), truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                        except Exception as e:
                            print(e)
                            pdb.set_trace()
                    else:
                        data = data.map(lambda x: self.tokenizer(list(zip(*[x[k] for k in key])), truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                    data = self.rename_cols(data, prefix[0])
                    if self.two_side:
                        if self.additional_cls:
                            try:
                                data = data.map(lambda x:  self.tokenizer(list(zip(*[map(self.appender, x[k]) if k == key[1] else x[k] for k in key[::-1]])), truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                            except:
                                pdb.set_trace()
                        else:
                            data = data.map(lambda x: self.tokenizer(list(zip(*[x[k] for k in key[::-1]])), truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                        data = self.rename_cols(data, prefix[1])
                else:
                    if self.additional_cls:
                        try:
                            data = data.map(lambda x: self.tokenizer(list(map(self.appender, x[key])), truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                        except:
                            pdb.set_trace()
                    else:
                        data = data.map(lambda x: self.tokenizer(x[key], truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                    data = self.rename_cols(data, prefix[0])
                    if self.two_side:
                        if self.additional_cls:
                            data = data.map(lambda x: self.tokenizer(list(map(self.appender, x[key])), truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                        else:
                            data = data.map(lambda x: self.tokenizer(x[key], truncation=True, max_length=self.max_len, padding='max_length'), batched=True)
                        data = self.rename_cols(data, prefix[1])
            except Exception as e:
                print(e)
                pdb.set_trace()
        data = self._set_format(data)
        return data

    def _reverse_keys(self, data):
        keys = self.tags[self.dataset.split('-')[0]][0]
        if len(keys) == 2:
            data = data.rename_column(keys[0], 'tempname')
            data = data.rename_column(keys[1], keys[0])
            data = data.rename_column('tempname', keys[1])
        else:
            print('Unable to reverse this data since number of input column is not 2')
        return data

    def setup(self, stage=None, datasets=None):
        if stage == 'fit' or stage is None:
            data = self.get_dataset(self.dataset, 'fit')
            self.data_train, self.data_val = data[0], data[1]
            if self.append_multiple_train:
                if datasets is None:
                    print('Append multiple flag is True, however no dataset is added')
                else:
                    for ds in datasets:
                        concat_data = self.add_dataset(ds, data={'train': self.data_train, 'val': self.data_val})
                        self.data_train = concat_data['train']
                        self.data_val = concat_data['val']
            if self.augment_reverse:
                if len(self.tags[self.dataset.split('-')[0]]) == 1 and len(self.tags[self.dataset.split('-')[0]][0]) == 2:
                    self.data_train = self.double_data(self.data_train)
                else:
                    print('Cannot reverse even with -augment_reverse = True')
            self.data_train = self._apply_tokenizer(self.data_train)
            self.format_labels = []
            self.data_val = self._apply_tokenizer(self.data_val)

        if stage == 'test' or stage is None:
            data = self.get_dataset(self.dataset, 'test')
            self.data_test = data[0]
            if self.append_multiple_test:
                if datasets is None:
                    print('Append multiple flag is True, however no dataset is added')
                else:
                    for ds in datasets:
                        concat_data = self.add_dataset(ds, data={'test': self.data_test})
                        self.data_test = concat_data['test']
            self.format_labels = []
            if self.eval_reverse:
                self.data_test = self._reverse_keys(self.data_test)
            self.data_test = self._apply_tokenizer(self.data_test)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.train_batch_size, shuffle=True, num_workers=0)

    # see pytorch-lightning tutorial to make this part better -> specifically dataset fetching
    # Done
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.eval_batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.eval_batch_size, num_workers=0)

    @classmethod
    def add_data_specific_args(cls, args):
        return args.get_data_module_args()
