# -*- coding: utf-8 -*-
import numpy as np
import itertools
from tqdm import tqdm
class kronbinations():
    # A class for scanning parameter landscapes
    # creates iterators over multiple parameters via kronecker products (itertools product), 
    # automatizes the construction of arrays to sotore results on the landscape, simplifies indexing 
    # while keeping the ability to add functions that get only executed in a specific subloop by tracking when a variable is changed

    def __init__(self, *values):
        # If values is a dictionary, also store the keys, so that change can be outputted by key
        if isinstance(values[0], dict):
            if len(values) > 1:
                raise VariableError('If values is a dictionary, it must be the only argument')
            self.keys = list(values[0].keys())
            values = list(values[0].values())
            self.return_as_dict = True
        else:
            self.keys = None
            self.return_as_dict = False
        # Define variables and iterators
        self.values_var = values
        self.lengths = [len(v) for v in self.values_var]
        n = len(self.lengths)
        self.ndims = n
        self.index_list = [np.arange(len(v)) for v in self.values_var]
        self.total_length = np.prod(self.lengths)
        # Define the iterators
        self.setup_iterator()

        self.do_index = False
        self.do_change = False
        self.do_tqdm = False

    def size(self):
        return self.total_length
    def shape(self):
        return tuple(self.lengths)
    def ndim(self):
        return self.ndims

    def empty(self, *var, **args):
        return np.empty(self.lengths, *var, **args)
    def ones(self, *var, **args):
        return np.ones(self.lengths, *var, **args)
    def zeros(self, *var, **args):
        return np.zeros(self.lengths, *var, **args)
    def full(self, *var, **args):
        return np.full(self.lengths, *var, **args)
    def random(self, *var, **args):
        return np.random.random(self.lengths, *var, **args)
    def randint(self, *var, **args):
        return np.random.randint(*var,  size=self.lengths, **args)
    def rng(self, rng_fun=np.random.default_rng().random, *var, **args):
        return rng_fun(*var, size=self.lengths, **args)

    def set(self, **args):
        key_substitution_list = [['index', 'do_index'], ['change', 'do_change'], ['progress', 'do_tqdm']]
        key_list = [v[0] for v in key_substitution_list]
        subs_list = [v[1] for v in key_substitution_list]
        for key, value in args.items():
            # Substitute certain keys from substitution list
            if key in key_list:
                key = subs_list[key_list.index(key)]
            if (key == 'return_as_dict' and value==True) and not isinstance(self.keys, list):
                raise ValueError('Keys are not defined, must create Object via dictionary in order to set "return_as_dict = True".')
            else:
                setattr(self, key, value)
    def get(self, *args):
        key_substitution_list = [['index', 'do_index'], ['change', 'do_change'], ['progress', 'do_tqdm']]
        key_list = [v[0] for v in key_substitution_list]
        subs_list = [v[1] for v in key_substitution_list]
        x = []
        for key in args:
            if key in key_list:
                key = subs_list[key_list.index(key)]
            x.append(getattr(self, key))
        return x

    def setup_iterator(self):
        self.product = itertools.product(*self.values_var)
        self.indexes = itertools.product(*self.index_list)

        last_indexes = -np.ones(self.ndims, dtype=int)
        last_values = [v[0] for v in self.values_var]
        changed_var = np.zeros(self.ndims, dtype=bool)
        if self.return_as_dict:
            self.last_values = dict(zip(self.keys, last_values))
            self.last_indexes = dict(zip(self.keys, last_indexes))
            self.changed_var = dict(zip(self.keys, changed_var))
        else:   
            self.last_values = last_values
            self.last_indexes = last_indexes
            self.changed_var = changed_var

    def __next__(self):
        last_values = next(self.product)
        curr_index = next(self.indexes)
        changed_var = tuple(np.not_equal(curr_index, self.last_indexes))
        if self.do_tqdm:
            self.loop.update(1)
        if self.return_as_dict:
            self.last_values = dict(zip(self.keys, last_values))
            self.last_indexes = dict(zip(self.keys, curr_index))
            self.changed_var = dict(zip(self.keys, changed_var))
        else:   
            self.last_values = last_values
            self.last_indexes = curr_index
            self.changed_var = changed_var
        return self.last_values, self.last_indexes, self.changed_var

    def kronprod(self, **args):
        self.set(**args)
        if self.do_tqdm:
            self.loop = tqdm(range(self.total_length))
        if self.do_index:
            if self.do_change:
                for i in range(self.total_length):
                    v,i,c = next(self)
                    yield i, v, c
            else:
                for i in range(self.total_length):
                    v,i,_ = next(self)
                    yield i, v
        else:
            if self.do_change: 
                for i in range(self.total_length):
                    v,_,c = next(self)
                    yield v, c
            else:
                for i in range(self.total_length):
                    v,_,_ = next(self)
                    yield v
        if self.do_tqdm:
            self.loop.close()
        self.setup_iterator()

    def changed(self, elem=None):
        if elem is None:
            return self.changed_var
        elif isinstance(elem, int):
            return self.changed_var[elem]
        elif isinstance(elem, str): # Outputs changed by key
            if isinstance(self.keys, list):
                return self.changed_var[elem]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')
                
    def index(self, elem=None):
        if elem is None:
            return self.last_indexes
        elif isinstance(elem, int):
            return self.last_indexes[elem]
        elif isinstance(elem, str): # By key
            if isinstance(self.keys, list):
                return self.last_indexes[elem]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')

    def value(self, elem=None):
        if elem is None:
            if self.return_as_dict:
                return dict(zip(self.keys, self.last_values))
            else:
                return self.last_values
        elif isinstance(elem, int):
            return self.last_values[elem]
        elif isinstance(elem, str):
            if isinstance(self.keys, list):
                return self.last_values[elem]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')