import os
import json
import logging
from tqdm import tqdm
import pdb as pdb
from glob import glob
from src.utils.word_embedding import read_word_embedding, load_word_embedding
from gensim import models
from datasets import Dataset

from attrdict import AttrDict

import numpy as np

import torch
import re

import src.constants as C


class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def __init__(self, expt_name):
        super(ContextFilter, self).__init__()
        self.expt_name = expt_name

    def filter(self, record):
        record.expt_name = self.expt_name
        return True


def get_logger(name, expt_name, log_format, logging_level, log_file_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.addFilter(ContextFilter(expt_name))

    return logger


def print_log(logger, dict):
    str = ''
    for key, value in dict.items():
        str += '{}: {}\t'.format(key.replace('_', ' '), value)
    str = str.strip()
    logger.info(str)


# For pytorch
def gpu_init_pytorch(gpu=False):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def check_nan(model, name):
    for p in model.parameters():
        norm = p.norm()
        if torch.isnan(norm):
            pdb.set_trace()


def check_gradients(model, name):
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            norm = p.grad.norm()
            if torch.isnan(norm):
                pdb.set_trace()
                return True
    return False


def initialize_embeddings(embedding, embed_path, embed_file,
                          w2id, device, vectype='glove'):

    '''
    Args:
        -embedding : embedding module
        -embed_file : file containing the pretrained embeddings
        -w2id : Vocabulary word2tokenid
    '''

    id2w = {}
    for k, v in w2id.items():
        id2w[v] = k

    if vectype == 'glove':
        if not os.path.exists(os.path.join(embed_path, embed_file)):
            read_word_embedding(embed_path, embed_file[:-4], embed_file)
        else:
            pass
        word_embeddings = load_word_embedding(embed_path, embed_file)
        kw = list(word_embeddings.keys())[0]
    elif vectype == 'word2vec':
        word_embeddings = models.KeyedVectors.load_word2vec_format(os.path.join(embed_path,
                                                                                embed_file),
                                                                   limit=200000, binary=True)
        kw = 'the'
    else:
        print('Undefined vectype, Exiting!!')
        sys.exit()

    words = list(w2id.keys())
    embed_dim = word_embeddings[kw].shape[-1]
    embedding_mat = np.zeros((len(words), embed_dim))

    for i, word in id2w.items():

        if word == 'EOS':
            embedding_mat[i] = torch.zeros(embed_dim)
        elif word not in word_embeddings:
            embedding_mat[i] = np.random.randn(embed_dim)*np.sqrt(2/(len(words) + embed_dim))
        else:
            embedding_mat[i] = word_embeddings[word]

    embedding.weight.data = torch.FloatTensor(embedding_mat).to(device)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\?", " ? ", string)

    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def accuracy_with_logits(logits, labels):
    _, preds = torch.max(logits, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

def accuracy(preds, labels):
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

def get_ckpt_args(file_path):
    hyperparamsfile = os.path.join(os.path.split(file_path)[0], 'hyperparams.json')
    with open(hyperparamsfile, 'r') as f:
        ckpt_args = json.load(f)
    ckpt_args = AttrDict(ckpt_args)
    return ckpt_args


# code to load or save model
def load_checkpoint(model, optimizer, file_path, logger, device, get_epacc=False):
    train_loss = None
    val_loss = None
    epoch = None
    val_acc = None

    if 'pth' not in os.path.split(file_path)[-1]:
        file_path = get_latest_checkpoint(file_path, logger)

    try:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['modelsd'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        val_acc = checkpoint['val_acc']
        try:
            outer_it = checkpoint['outer_it']
        except Exception:
            outer_it = '#'

        model.to(device)
        # optimizer.to(device)

        logger.info('Successfully loaded checkpoint from {}'.format(file_path))
        string = "Outer it [{}] - Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc:{:.4f}"
        logger.info(string.format(outer_it,
                                  epoch,
                                  train_loss,
                                  val_loss,
                                  val_acc))

        if get_epacc:
            return epoch, train_loss, val_loss, val_acc

    except Exception as e:
        logger.info('{} \n No checkpoint found on {}'.format(e, file_path))


def save_checkpoint(args, state, prev_ckpts, logger, classifier=True):
    ep = state['epoch']
    outer_it = state['outer_it']
    if classifier:
        namestr = args.classifier
    else:
        namestr = args.paraphraser
    dir_path = os.path.join(C.MODELPATH, args.dataset, '{}-{}'.format(namestr, args.postfix))
    os.makedirs(dir_path, exist_ok=True)

    try:
        save_path = os.path.join(dir_path, 'model-{}-{}.pth'.format(outer_it, ep))
        try:
            with open(os.path.join(dir_path, 'hyperparams.json'), 'w') as f:
                json.dump(vars(args), f)
        except:
            with open(os.path.join(dir_path, 'hyperparams.json'), 'w') as f:
                json.dump(args, f)

        torch.save(state, save_path)
        logger.info('Model saved successfully at {}'.format(save_path))
    except Exception as e:
        pdb.set_trace()
        logger.info('Model save unsuccessful. Please debug. Running training normally...')
        return prev_ckpts

    prev_ckpts.append((ep, outer_it))
    if len(prev_ckpts)>args.keep_ckpts:
        del_ep = prev_ckpts[0][0]
        del_ot = prev_ckpts[0][1]
        prev_ckpts = prev_ckpts[-args.keep_ckpts:]

        if del_ot == outer_it:
            del_path = os.path.join(dir_path, 'model-{}-{}.pth'.format(del_ot, del_ep))
            try:
                os.remove(del_path)
            except Exception:
                pass
            logger.info('Removed obsolete model checkpoint: {}'.format(del_path))

    return prev_ckpts

def remove_obsolete_checkpoint(args, outer_it, logger, classifier=True):
    if classifier:
        namestr = args.classifier
    else:
        namestr = args.paraphraser
    dir_path = os.path.join(C.MODELPATH, args.dataset, '{}-{}'.format(namestr, args.postfix))
    ckpt_names = glob('{}/model-{}-*.pth'.format(dir_path, outer_it))
    for ckpt in ckpt_names:
        try:
            os.remove(ckpt)
        except Exception:
            pass
        logger.info('Removed Previous iterations checkpoint : {}'.format(ckpt))

def get_ckpt_name(args, ep, outer_it, classifier=True):
    if classifier:
        namestr = args.classifier
    else:
        namestr = args.paraphraser

    dir_path = os.path.join(C.MODELPATH, args.dataset, '{}-{}'.format(namestr, args.postfix))
    return os.path.join(dir_path, 'model-{}-{}.pth'.format(outer_it, ep))

def get_latest_checkpoint(file_path, logger):
    logger.info('Trying to fetch the latest checkpoint from {}'.format(file_path))
    ckpt_names = glob('{}/*.pth'.format(file_path))

    if len(ckpt_names) == 0:
        logger.info('No checkpoints found in {}'.format(file_path))
    else:
        latest_eps = max([int(os.path.split(fname)[-1].split('-')[-1].split('.')[0]) for fname in ckpt_names])
        logger.info('Checkpoint found with epoch : {}'.format(latest_eps))
        prefix = os.path.split(ckpt_names[0])[0]
        postfix = 'model-{}.pth'.format(latest_eps)
        file_path = os.path.join(prefix, postfix)
    return file_path

def convert_enc_to_string(input_ids, tokenizer, skip_special_tokens=True):
    string = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids,
                                                                                skip_special_tokens=skip_special_tokens))
    return string.strip()

MAPSCHEMA = {
    'cola': ['idx', 'label', 'sentence'],
    'sst2': ['idx', 'label', 'sentence'],
    'mrpc': ['idx', 'label', 'sentence1', 'sentence2'],
    'stsb': ['idx', 'label', 'sentence1', 'sentence2'],
    'rte': ['idx', 'label', 'sentence1', 'sentence2'],
    'wnli': ['idx', 'label', 'sentence1', 'sentence2'],
    'qqp': ['idx', 'label', 'question1', 'question2'],
    'qnli': ['idx', 'label', 'question', 'sentence'],
    'trec': ['idx', 'label-coarse', 'label-fine', 'text']
}

def create_dataspecific_dict(s1, s2, labels, dataset, c):
    dictfinal = {}
    dataset = dataset.split('-')[0]

    dictfinal['idx'] = [next(c) for _ in range(len(s1))]

    if dataset in ['sst2', 'cola']:
        dictfinal['label'] = labels
        dictfinal['sentence'] = s1
    elif dataset in ['mrpc', 'stsb', 'rte', 'wnli']:
        dictfinal['label'] = labels
        dictfinal['sentence1'] = s1
        dictfinal['sentence2'] = s2
    elif dataset in ['qqp']:
        dictfinal['label'] = labels
        dictfinal['question1'] = s1
        dictfinal['question2'] = s2
    elif dataset in ['qnli']:
        dictfinal['label'] = labels
        dictfinal['question'] = s1
        dictfinal['sentence'] = s2
    elif dataset in ['trec']:
        dictfinal['label-coarse'] = labels
        dictfinal['label-fine'] = labels
        dictfinal['text'] = s1
    else:
        raise NotImplementedError
        pdb.set_trace

    augmented_set = Dataset.from_dict(dictfinal)
    return augmented_set

def get_sentences_appended(dataset_class, dataset_dict, outer_it, typedata='selected'):
    curr_dataloader = dataset_class.get_dataloader(dataset_dict, typedata)
    listdictfinal = []
    for i, batch in enumerate(tqdm(curr_dataloader, 'Creating dictionary for analysis of {} points'.format(typedata))):
        k = list(batch.keys())[0]
        for i, _ in enumerate(batch[k]):
            dictfinal = {}
            dictfinal['outer_it'] = outer_it
            dictfinal['typedata'] = typedata
            for k in batch:
                try:
                    dictfinal[k] = batch[k][i].item()
                except:
                    dictfinal[k] = batch[k][i]
            listdictfinal.append(dictfinal)


    return listdictfinal
