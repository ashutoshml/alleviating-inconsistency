import os
import json
from tqdm import tqdm
import pdb as pdb
from itertools import count

from datasets import load_dataset, concatenate_datasets, Dataset
from pytorch_lightning import seed_everything

from args import Args
import constants as C


def loadset(dataset):
    if dataset in ['sst2', 'qqp', 'cola', 'stsb', 'rte', 'wnli', 'mrpc', 'qnli']:
        return load_dataset('glue', dataset)
    if dataset == 'snli':
        return load_dataset('snli')
    if dataset == 'trec':
        return load_dataset('trec')
    else:
        raise ValueError('Incorrect dataset type')


def write_train_to_train_test_file(args, trainfile, testfile, fulldataset, labels):
    fulldataset = fulldataset['train'].shuffle(seed=args.seed)

    if args.dataset == 'stsb':
        fraction = int(args.fraction*len(fulldataset))
        testset = Dataset.from_dict(fulldataset[:fraction])
        trainset = Dataset.from_dict(fulldataset[fraction:])
    else:
        if args.equal_testset:
            if args.dataset == 'trec':
                splits = [fulldataset.filter(lambda ex: ex['label-coarse'] == lab) for lab in labels]
            else:
                splits = [fulldataset.filter(lambda ex: ex['label'] == lab) for lab in labels]
            lensplits = [len(s) for s in splits]
            fractionmin = int(args.fraction*min(lensplits))
            testsets = [Dataset.from_dict(s[:fractionmin]) for s in splits]
            trainsets = [Dataset.from_dict(s[fractionmin:]) for s in splits]
            trainset = concatenate_datasets(trainsets)
            testset = concatenate_datasets(testsets)
        else:
            fraction = int(args.fraction*len(fulldataset))
            testset = Dataset.from_dict(fulldataset[:fraction])
            trainset = Dataset.from_dict(fulldataset[fraction:])

    with open(trainfile, 'w') as writef:
        for ix, data in enumerate(tqdm(trainset)):
            dv = json.dumps({key: trainset[ix][key] for key in C.SCHEMA[args.dataset]})
            writef.write(dv + '\n')

    with open(testfile, 'w') as writef:
        for ix, data in enumerate(tqdm(testset)):
            dv = json.dumps({key: testset[ix][key] for key in C.SCHEMA[args.dataset]})
            writef.write(dv + '\n')


def write_val_to_file(args, valfile, val_dataset):
    with open(valfile, 'w') as writef:
        for ix, data in enumerate(tqdm(val_dataset)):
            dv = json.dumps({key: val_dataset[ix][key] for key in C.SCHEMA[args.dataset]})
            writef.write(dv + '\n')


def construct_idx(fulldataset):
    line = count(-1)
    fulldataset = fulldataset.map(lambda ex: {'idx': next(line)})
    return fulldataset


def main(args):
    fulldataset = loadset(args.dataset)
    valset = fulldataset['validation']

    if 'idx' not in fulldataset['train'].features:
        fulldataset = construct_idx(fulldataset)
    if args.dataset == 'trec':
        labels = set(fulldataset['train']['label-coarse'])
    else:
        labels = set(fulldataset['train']['label'])

    if args.equal_testset:
        destinationdir = os.path.join(C.DATAPATH_CLASS, '{}-eq'.format(args.dataset))
    else:
        destinationdir = os.path.join(C.DATAPATH_CLASS, '{}-new'.format(args.dataset))

    os.makedirs(destinationdir, exist_ok=True)

    trainfile = os.path.join(destinationdir, 'train.json')
    validationfile = os.path.join(destinationdir, 'validation.json')
    testfile = os.path.join(destinationdir, 'test.json')

    if args.overwrite_traintest:
        print('Writing Training and Validation Files')
        write_train_to_train_test_file(args, trainfile, testfile, fulldataset, labels)
    elif os.path.exists(trainfile) and os.path.exists(validationfile):
        print('File present in the location')
    else:
        print('Some files missing. Try with --overwrite_traintest')

    write_val_to_file(args, validationfile, valset)

    print('File writing Completed')


if __name__ == "__main__":
    args = Args()

    parser = args.get_generic_args()
    parser = args.get_splitter_set_args()

    args = parser.parse_args()

    seed_everything(args.seed)

    main(args)
