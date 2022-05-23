from collections import OrderedDict
import os
import json
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('-modeldir', required=True)
args = parser.parse_args()


exclude_ids = set([])

src_file = "src/create_references.py"

_dataset = ['paws', 'sst2-eq', 'sst2-new', 'rte-eq', 'mrpc-new', 'qnli-eq', 'qqp-new']
_ckpt = [os.path.join(args.modeldir, name) for name in os.listdir(args.modeldir) if os.path.isdir(os.path.join(args.modeldir, name)) and 'Incorrect' not in os.path.join(args.modeldir, name)]

i, count = 0, 0

set_configs = set()

for ckpt in _ckpt:
    for dataset in _dataset:
        config = OrderedDict()
        config['src_file'] = src_file
        config['ckpt'] = ckpt
        config['dataset'] = dataset
        config['ebs'] = 512

        if i not in exclude_ids:
            set_configs.add(json.dumps(config))
        i += 1

with open("precommands.json", 'w') as f:
    for s in set_configs:
        count += 1        
        f.write(s + "\n")
        print("Inserting {}".format(count), end="\r")

print('\nInserted {} in precommands.json. Complete'.format(count))
