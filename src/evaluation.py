import os
import sys
import pdb as pdb
import numpy as np
from scipy import stats

from args import Args

import constants as C
import pandas as pd
from datasets import load_metric, load_dataset

from prettytable import PrettyTable


class Table(object):
    def __init__(self):
        self.metric = ['accuracy', 'f1', 'matthews_correlation', 'pearsonr', 'spearmanr', 'mse', 'precision', 'recall']
        self.table = PrettyTable(["Seed", "Model", "Dual", "Consistent", "Additional_CLS", "Reverse", "Divergence", "Dataset", "Combination", "Columns"] + self.metric)
        self.alldict = {"Seed":[], "Model": [], "Dual": [], "Consistent": [], "Additional_CLS": [], "Reverse": [], "Divergence": [], "Dataset": [], "Combination": [], "Columns": []}
        for m in self.metric:
            self.alldict[m] = []

    def _set_configuration(self, filename):
        divergence = 'None'
        if 'divergence=js' in filename:
            divergence = 'js'
        elif 'divergence=kl' in filename:
            divergence = 'kl'
        self.configuration = {
            'Seed': int(filename.split('seed')[-1].split('-')[0][1:]) if 'seed' in filename else -1,
            'Model': 'roberta' if 'roberta' in filename else 'bert',
            'Dual': True if 'dual' in filename else False,
            'Consistent': True if 'consistency=True' in filename else False,
            'Additional_CLS': True if 'additional_cls=True' in filename else False,
            'Reverse': True if 'reverse=True' in filename else False,
            'Divergence': divergence
        }

    def add_configuration_dict(self):
        for key in self.configuration:
            self.alldict[key].append(self.configuration[key])

    def update_dict(self, dataset, combination, columns, metric_list):
        self.add_configuration_dict()
        self.alldict["Dataset"].append(dataset)
        self.alldict["Combination"].append(' -> '.join(combination))
        self.alldict["Columns"].append(columns)
        for i, m in enumerate(self.metric):
            self.alldict[m].append(metric_list[i])

    def conv_to_dict(self, results):
        dict_final = {}
        for res in results:
            for key, value in res.items():
                dict_final[key] = value
        return dict_final

    def insert_entry(self, dataset, combination, columns, results):
        metric_list = []
        results = self.conv_to_dict(results)
        for metric in self.metric:
            metric_list.append(results[metric] if metric in results else '-')
        list_entry = [self.configuration[k] for k in ['Seed', 'Model', 'Dual', 'Consistent', 'Additional_CLS', 'Reverse', 'Divergence'] ]
        list_entry += [dataset, ' -> '.join(combination), columns]
        list_entry += metric_list
        self.update_dict(dataset, combination, columns, metric_list)
        self.table.add_row(list_entry)

    def print_table(self):
        print(self.table)

    def save_table(self, filename):
        df = pd.DataFrame(self.alldict)
        print('Saving Results in file: {}'.format(filename))
        df.to_csv(filename, sep='\t', encoding='utf-8', index=False)


class MSE:
    def __init__(self):
        pass

    def compute(self, predictions=None, references=None):
        return {'mse': np.square(np.subtract(predictions, references)).mean()}


class FallbackPearsonR:
    def compute(self, predictions=None, references=None):
        return {'pearsonr': stats.pearsonr(predictions, references)[0]}

class FallbackSpearmanR:
    def compute(self, predictions=None, references=None):
        return {'spearmanr': stats.spearmanr(predictions, references)[0]}

class Evaluator:
    def __init__(self, args, table):
        self.table = table
        self.config = args

        self.evaluate_single = False
        if self.config.references is not None and self.config.predictions is not None:
            print('Evaluation Initialized!')
            self.evaluate_single = True
        elif self.config.pretrain_path is not None:
            print('Evaluation Initialized!')
        else:
            print('No evaluation possible - Missing necessary files. Exiting!')
            sys.exit()

        self.metric_type = {
            'prediction': 'classification',
            'confidence': 'regression',
            'score': 'regression'
        }

        self.metric_all = {'regression': ['pearsonr', 'spearmanr', 'mse']}
        self.incorrect_examples = {}

    def load_metric(self, dataset=None, metric=None, task='classification'):
        if dataset is not None:
            return load_metric('glue', dataset.split("-")[0])
        elif metric is not None:
            if metric == 'mse':
                return MSE()
            return load_metric(metric)
        else:
            if task == 'classification':
                return load_metric('accuracy')
            else:
                return load_metric('spearmanr')

    def verify_arguments(self):
        if self.config.pretrain_path is None:
            if self.config.ground_truth is None or args.prediction is None:
                print('-ref or -pred missing!')
                return False

        return True

    def print_results(self, results):
        table = PrettyTable(["Key", "Result"])
        for key, result in results.items():
            table.add_row([key, result])
        print(table)

    def evaluate(self, predictions, references, allmetric, metric_type='classification'):
        ans = []
        for key, metric in allmetric.items():
            if metric_type == 'regression' and key not in self.metric_all[metric_type]:
                continue
            try:
                computed_val = metric.compute(predictions=predictions, references=references)
            except:
                computed_val = {}
            for ckey, value in computed_val.items():
                if np.isnan(value) and metric_type == 'regression':
                    if ckey == 'pearsonr':
                        metric = FallbackPearsonR()
                    elif ckey == 'spearmanr':
                        metric = FallbackSpearmanR()
                    else:
                        metric = metric
                    try:
                        computed_val[ckey] = metric.compute(predictions=predictions, references=references)[ckey]
                    except:
                        pass
            ans.append(computed_val)
        return ans

    def get_all_metrics(self, metrics=None, currdataset=None):
        if metrics == 'all' or (metrics is None and currdataset is None):
            return {m: self.load_metric(metric=m) for m in C.ALL_METRIC}
        if args.metric is not None:
            allmetric = {m: self.load_metric(metric=m) for m in metrics}
        else:
            try:
                allmetric = {currdataset: self.load_metric(dataset=currdataset)}
            except:
                return {m: self.load_metric(metric=m) for m in C.ALL_METRIC}
        return allmetric

    def get_dataframe_cols_from_files(self, predictions=None, references=None):
        try:
            predictions_df = pd.read_csv(predictions, sep='\t')
            cols = list(predictions_df.columns)
            references_df = pd.read_csv(references, sep='\t')
            if 'confidence' in cols:
                predictions_df = self.convert_confs_to_one(predictions_df)
                references_df = self.convert_confs_to_one(references_df)

            return predictions_df, references_df, cols
        except:
            return None, None, None

    def convert_confs_to_one(self, dataframe=None):
        dataframe.loc[dataframe['prediction'] == 0, 'confidence'] = 1. - dataframe.loc[dataframe['prediction'] == 0, 'confidence']
        return dataframe

    def save_incorrect_examples(self, folder, predictions_df, references_df):
        dataset = os.path.split(os.path.split(folder)[0])[1].split('-')[0]
        if dataset in ['qqp', 'mrpc']: dataset = '{}-new'.format(dataset)
        datadir = os.path.join(C.DATAPATH_CLASS, dataset)
        # CACHED = os.path.join(C.CACHE_DIR['datasets'], dataset)
        filen = os.path.join(self.config.pretrain_path, 'Incorrect', '{}-{}-incorrect_ex.csv'.format('-'.join(folder.split('/')), dataset))

        if 'qqp' in dataset or 'mrpc' in dataset:
            data = load_dataset('json', data_files={'test': os.path.join(datadir, 'test.json')
                                                               }, split=['test'])
            data = data[0]
        else:
            data = load_dataset('paws', 'labeled_final')
            data = data['test']

        dict_incorrect = {
                'sentence1':[],
                'sentence2':[],
                'label': [],
                'r2l-c': [],
                'r2l-p': [],
                'l2r-c': [],
                'l2r-p': []
        }

        for i in range(len(predictions_df['prediction'])):
            if predictions_df['prediction'][i] != references_df['prediction'][i] or (predictions_df['confidence'][i] - references_df['confidence'][i]) > 0.2:
                dict_incorrect['sentence1'].append(data[i]['sentence1'] if 'qqp' not in dataset else data[i]['question1'])
                dict_incorrect['sentence2'].append(data[i]['sentence2'] if 'qqp' not in dataset else data[i]['question2'])
                dict_incorrect['label'].append(data[i]['label'])
                dict_incorrect['r2l-c'].append(predictions_df['confidence'][i])
                dict_incorrect['r2l-p'].append(predictions_df['prediction'][i])
                dict_incorrect['l2r-c'].append(references_df['confidence'][i])
                dict_incorrect['l2r-p'].append(references_df['prediction'][i])

        incorrect_ex = pd.DataFrame(dict_incorrect)
        print('Saving incorrect examples to file: {}'.format(filen))
        incorrect_ex.to_csv(filen, sep='\t', encoding='utf-8', index=False)

    def run_evaluation(self):
        if self.evaluate_single:
            allmetric = self.get_all_metrics()
            predictions_df, references_df, cols = self.get_dataframe_cols_from_files(args.predictions, args.references)
            if predictions_df is None:
                print('Files missing - exiting')
                sys.exit()
            ans = {}
            for c in cols:
                ans[c] = self.evaluate(predictions_df[c], references_df[c], allmetric, self.metric_type[c])
            self.print_results(ans)

        elif self.config.pretrain_path is not None:
            for ppath in os.listdir(self.config.pretrain_path):
                pretrain_path = os.path.join(self.config.pretrain_path, ppath)
                if not os.path.isdir(pretrain_path) or 'Incorrect' in pretrain_path:
                    continue
                finetunedir = os.path.join(pretrain_path, 'finetune')
                topfolders = [os.path.join(finetunedir, name, 'classification') for name in os.listdir(finetunedir) if os.path.isdir(os.path.join(finetunedir, name))]

                combinations = [('results-rev-False.csv', 'references.csv'),
                                ('results-rev-True.csv', 'references.csv'),
                                ('results-rev-True.csv', 'results-rev-False.csv')]


                for folder in topfolders:
                    self.table._set_configuration(folder)
                    currdataset = os.path.split(os.path.split(folder)[0])[1].split('-')[0]
                    if 'qqp' in currdataset or 'paws' in currdataset or 'mrpc' in currdataset:
                        combinations.append(('results-cons-True-rev-True.csv', 'results-cons-True-rev-False.csv'))
                        combinations.append(('results-cons-True-rev-False.csv', 'references.csv'))
                        combinations.append(('results-cons-True-rev-True.csv', 'references.csv'))

                    if args.dataset is not None and currdataset not in args.dataset:
                        continue

                    for comb in combinations:
                        predictions_df, references_df, cols = self.get_dataframe_cols_from_files(os.path.join(folder, comb[0]),
                                                                                                      os.path.join(folder, comb[1]))
                        if predictions_df is None:
                            continue

                        if comb[0] == 'results-cons-True-rev-True.csv' and comb[1] == 'results-cons-True-rev-False.csv':
                            self.save_incorrect_examples(folder, predictions_df, references_df)

                        ans = {}
                        print('Dataset: {} | Combination : {}'.format(currdataset, comb))
                        for c in cols:
                            if self.metric_type[c] == 'classification':
                                allmetric = self.get_all_metrics(currdataset=currdataset)
                            else:
                                allmetric = self.get_all_metrics()
                            ans[c] = self.evaluate(predictions_df[c], references_df[c], allmetric, self.metric_type[c])
                            self.table.insert_entry(currdataset, comb, c, ans[c])
                        self.print_results(ans)

            self.table.print_table()
            print('Saving Table ...')
            self.table.save_table(os.path.join(self.config.pretrain_path, self.config.save_file))
        else:
            raise NotImplementedError

    @classmethod
    def add_evaluator_args(cls, args):
        return args.get_evaluation_args()

if __name__ == '__main__':
    args = Args()

    parser = args.get_generic_args()
    parser = Evaluator.add_evaluator_args(args)

    args = parser.parse_args()

    table = Table()
    evaluator = Evaluator(args, table)
    if not evaluator.verify_arguments():
        print('Exiting evaluation')
        sys.exit()
    evaluator.run_evaluation()
