import os
from src.helper import *
import src.constants as C
import pdb as pdb

# Steps
# 1. Create temp file - in DATAPATHDIR, dataset, test.json
# 2. parse dataset into format - check src.constants
# 3. Write into temp files - created in 1
# 4. return get_custom_data

class UserData:
    def __init__(self, filename, dataset, logger):
        self.filename = filename
        self.dataset = dataset
        self.logger = logger
        self.save_path = os.path.join(C.DATAPATH_CLASS, '{}-usertmp'.format(dataset))
        os.makedirs(self.save_path, exist_ok=True)
        self.data = self.read_file()
        self.convert_to_fmt()

    def read_file(self):
        data = []
        if self.dataset.split('-')[0] in C.SINGLE_TAB:
            with open(self.filename, 'r') as f:
                for line in f:
                    data.append(line.strip())
        else:
            with open(self.filename, 'r') as f:
                for line in f:
                    s0, s1 = line.strip().split('\t')
                    data.append((s0, s1))
        return data

    def convert_to_fmt(self):
        with open(os.path.join(self.save_path, 'test.json'), 'w') as writef:
            keys_schema = C.SCHEMA[self.dataset.split('-')[0]]
            for i, line in enumerate(self.data):
                if self.dataset.split('-')[0] in C.SINGLE_TAB:
                    try:
                        if self.dataset.split('-')[0] != 'trec':
                            dict_ans = {
                                keys_schema[0]: i,
                                keys_schema[1]: -1,
                                keys_schema[2]: line.strip()
                            }
                        else:
                            dict_ans = {
                                keys_schema[0]: i,
                                keys_schema[1]: -1,
                                keys_schema[2]: -1,
                                keys_schema[3]: line.strip()
                            }
                    except Exception as e:
                        self.logger.info('Incompatible type of data provided. Kindly check format')

                else:
                    try:
                        dict_ans = {
                            keys_schema[0]: i,
                            keys_schema[1]: -1,
                            keys_schema[2]: line[0],
                            keys_schema[3]: line[1]
                        }
                    except Exception as e:
                        self.logger.info('Incompatible type of data provided. Kindly check format')

                dv = json.dumps(dict_ans)
                writef.write(dv + '\n')

    def get_file_name(self):
        return '{}-usertmp'.format(self.dataset)

    def del_temp(self):
        try:
            os.remove(self.save_path)
            self.logger.info('Removed temporary files')
        except:
            self.logger.info('Nothing to remove here')
