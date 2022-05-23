import argparse
import json
import subprocess
import os


def get_cmd(config_line):
    config = config_line
    
    cmd = '{}'.format(config['src_file'])
    src_file = config['src_file']
    del config['src_file']

    for key, value in config.items():
        if value is None:
            continue
        if type(value) == bool:
            if value:
                cmd += ' -{}'.format(key)
            continue
        cmd += ' -{} {}'.format(key, str(value))

    config['src_file'] = src_file
    return cmd, config


def cpu_run(args):
    all_cmd = []
    with open(args.input_file, 'r') as f:
        for line in f:
            config_line = json.loads(line)
            cmd, _ = get_cmd(config_line)
            cmd = 'TOKENIZERS_PARALLELISM=False python {}'.format(cmd)
            print('Command: {}'.format(cmd))
            cmd_run = cmd#.split()
            all_cmd.append(cmd_run)

    print("Using os.system command. Tried using subprocess but it was resulting in non-execution of command.")
    for cmd_run in all_cmd:
        os.system(cmd_run)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reference Creator')
    parser.add_argument("-i", "--input_file", default="precommands.json", help="File name where all json configurations are stored")
    args = parser.parse_args()
    cpu_run(args)
