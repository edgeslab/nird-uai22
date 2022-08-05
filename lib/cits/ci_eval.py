import os
import numpy as np
import pandas as pd
from uai2017experiments.utils import AUPC

class EvalCI():
    def __init__(self, a=0.05, names=[]):
        self.alpha = a
        self.beta = a
        self.names = names

        self.initialize()


    def initialize(self):
        self.p_vals_null = {}
        self.p_vals_alt = {}
        self.times_null = {}
        self.times_alt = {}

        for name in self.names:
            self.p_vals_null[name] = []
            self.p_vals_alt[name] = []
            self.times_null[name] = []
            self.times_alt[name] = []

        self.results = {'p_null' : [], 'p_alt' : [], 'type_i' : [], 'type_ii' : [], 'aupc': [], 'times_null': [], 'times_alt': []}


    def type_I(self, name):
        p_values = self.p_vals_null[name]
        return np.mean(np.array(p_values) < self.alpha)


    def type_II(self, name):
        p_values = self.p_vals_alt[name]
        return np.mean(np.array(p_values) > self.beta)


    def aupc(self, name):
        p_values = self.p_vals_alt[name]
        return AUPC(p_values)


    def __get_values(self, name, null):
        return self.p_vals_null[name] if null else self.p_vals_alt[name]


    def __get_times(self, name, null):
        return self.times_null[name] if null else self.times_alt[name]


    def log_p_vals(self, log_file):
        data = {}
        data['trials'] = range(len(self.p_vals_null[self.names[0]]))
        for name in self.names:
            data['%s_null' % name] = self.p_vals_null[name]
            data['%s_alt' % name] = self.p_vals_alt[name]
            data['%s_times' % name] = self.__get_times(name, True)
        df = pd.DataFrame(data)
        if(os.path.isfile(log_file)):
            df.to_csv(log_file, index=False, header=None, mode='a')
        else:
            df.to_csv(log_file, index=False)


    def load_p_vals(self, log_file):
        try:
            df = pd.read_csv(log_file)
            for name in self.names:
                self.p_vals_null[name] = list(df['%s_null' % name])
                self.p_vals_alt[name] = list(df['%s_alt' % name])
                self.times_null[name] = list(df['%s_times' % name])
                self.times_alt[name] = list(df['%s_times' % name])
        except FileNotFoundError:
            print('ERROR: log file %s doesn\'t exist!' % log_file)


    def add_result(self, name, p_val, null=True, time=0):
        p_values = self.__get_values(name, null)
        p_values.append(p_val)

        times = self.__get_times(name, null)
        times.append(time)


    def mean_values(self, name, null=True):
        p_values = self.__get_values(name, null)
        return np.mean(np.array(p_values))


    def total_time(self, name, null=True):
        times = self.__get_times(name, null)
        return np.sum(np.array(times))


    def gen_result(self):
        for name in self.names:
            self.results['p_null'].append(self.mean_values(name, null=True))
            self.results['p_alt'].append(self.mean_values(name, null=False))
            self.results['type_i'].append(self.type_I(name))
            self.results['type_ii'].append(self.type_II(name))
            self.results['aupc'].append(self.aupc(name))
            self.results['times_null'].append(self.total_time(name, null=True))
            self.results['times_alt'].append(self.total_time(name, null=False))
        self.rdf = pd.DataFrame(data=self.results, index=self.names)

        return self.rdf
            