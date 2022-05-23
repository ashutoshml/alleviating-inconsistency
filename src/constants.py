import os

WORD2VEC_PATH = os.path.join('data', 'wordvecs', 'word2vec')
WORD2VEC_FILE = 'GoogleNews-vectors-negative300.bin'

GLOVE_PATH = os.path.join('data', 'wordvecs', 'glove')
GLOVE_FILE = 'glove.6B.300d.txt-pkl'

LOGFMT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
DATEFMT = '%Y-%m-%d-%H:%M:%S'

MODELPATH = 'Models'
LOGPATH = 'Logs'

DATAPATH_CLASS = 'data/classification'
DATAPATH_GEN = 'data/generation'
DATAPATH_UNPROCESSED_GEN = 'data/unprocessed-generation'
DATAPATH_UNPROCESSED_CLASS = 'data/unprocessed-classification'

RESPATH = 'Classification'

NCLASS_DATASET = {
    'sst2': 2,
    'qnli': 3,
    'wnli': 3,
    'qqp': 2,
    'rte': 2,
    'trec': 6,
    'mrpc': 2,
    'stsb': 1,
    'cola': 2,
    'paws': 2
}

SCHEMA = {
    'cola': ['idx', 'label', 'sentence'],
    'sst2': ['idx', 'label', 'sentence'],
    'mrpc': ['idx', 'label', 'sentence1', 'sentence2'],
    'stsb': ['idx', 'label', 'sentence1', 'sentence2'],
    'rte': ['idx', 'label', 'sentence1', 'sentence2'],
    'wnli': ['idx', 'label', 'sentence1', 'sentence2'],
    'qqp': ['idx', 'label', 'question1', 'question2'],
    'qnli': ['idx', 'label', 'question', 'sentence'],
    'trec': ['idx', 'label-coarse', 'label-fine', 'text'],
    'paws': ['id', 'label', 'sentence1', 'sentence2']
}

GLUE_TASKS = ['cola', 'sst2', 'mrpc', 'stsb', 'rte', 'wnli', 'qqp', 'qnli']

SINGLE_TAB = ['cola', 'sst2', 'trec']
DOUBLE_TAB = ['mrpc', 'stsb', 'rte', 'wnli', 'qqp', 'qnli']

RENAME_COLS = {
        'paws': {
            'qqp': [('id', 'idx'), ('sentence1', 'question1'), ('sentence2', 'question2'), ('label', 'label')]
        },
        'mrpc': {
            'qqp':[('idx', 'idx'), ('sentence1', 'question1'), ('sentence2', 'question2'), ('label', 'label')]
        }
    }


MAX_LENGTH = {
    'cola': 150,
    'sst2': 150,
    'mrpc': 300,
    'stsb': 300,
    'rte': 300,
    'wnli': 300,
    'qqp': 300,
    'qnli': 300,
    'trec': 150
}


HIDDEN_DIM = {
    'bert': {
        'normal': 768
    },
    'roberta': {
        'normal': 768
    },
    'albert': {
        'normal': 1024
    }
}

CONFIG = {
        'bert': 'bert-base-cased',
        'roberta': 'roberta-base',
        'albert': 'albert-large-v2'
}

AVAILABLE_MODELS = ['bert', 'roberta', 'albert']
AVAILABLE_DATASETS = ['bert', 'roberta', 'albert']

SPECIAL_TOKENS = ['[CLS_PARA]']
ALL_METRIC = ['accuracy', 'f1', 'matthews_correlation', 'pearsonr', 'spearmanr', 'mse']
