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
        if isinstance(values, dict):
            self.keys = values.keys()
            values = values.values()
        else:
            self.keys = None
        # Define variables and iterators
        self.lengths = [len(v) for v in values]
        n = len(self.lengths)
        self.index_list = [np.arange(len(v)) for v in values]
        self.total_length = np.prod(self.lengths)
        self.product = itertools.product(*values)
        self.indexes = itertools.product(*self.index_list)
        self.last_indexes = -np.ones(n, dtype=int)
        self.last_values = [v[0] for v in values]
        self.changed = np.zeros(n, dtype=bool)

        self.do_index = False
        self.do_change = False
        self.do_tqdm = False
        self.return_as_dict = False

    def empty(self, **args):
        return np.empty(self.lengths, **args)
    def ones(self, **args):
        return np.ones(self.lengths, **args)
    def zeros(self, **args):
        return np.zeros(self.lengths, **args)
    def full(self, **args):
        return np.full(self.lengths, **args)
    def random(self, **args):
        return np.random.random(self.lengths, **args)
    def randint(self, **args):
        return np.random.randint(self.lengths, **args)
    def rng_random(self, rng=np.random.default_rng(), **args):
        return rng.random(self.lengths, **args)

    def set(self, **args):
        for key, value in args.items():
            setattr(self, key, value)

    def __next__(self):
        self.values = next(self.product)
        curr_index = next(self.indexes)
        self.changed = tuple(np.not_equal(curr_index, self.last_indexes))
        self.last_indexes = curr_index  
        if self.do_tqdm:
            self.loop.update(1)
        if self.return_as_dict:
            return dict(zip(self.keys, self.values)), curr_index, self.changed
        else:   
            return self.values, curr_index, self.changed

    def kronprod(self, index=False, change=False, progress=False, return_as_dict=False):
        self.do_index = index
        self.do_change = change
        self.do_tqdm = progress
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

    def changed(self, elem=None):
        if elem is None:
            return self.changed
        elif isinstance(elem, int):
            return self.changed[elem]
        elif isinstance(elem, str): # Outputs changed by key
            if isinstance(self.keys, list):
                ind = self.keys.index(elem)
                return self.changed[ind]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')
                
    def index(self, elem=None):
        if elem is None:
            return self.last_indexes
        elif isinstance(elem, int):
            return self.last_indexes[elem]
        elif isinstance(elem, str): # By key
            if isinstance(self.keys, list):
                ind = self.keys.index(elem)
                return self.last_indexes[ind]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')
    
    def index(self, elem=None):
        if elem is None:
            return self.last_indexes
        elif isinstance(elem, int):
            return self.last_indexes[elem]
        elif isinstance(elem, str): # By key
            if isinstance(self.keys, list):
                ind = self.keys.index(elem)
                return self.last_indexes[ind]
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
                ind = self.keys.index(elem)
                return self.last_values[ind]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')