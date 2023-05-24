# -*- coding: utf-8 -*-
import numpy as np
import itertools
from weighted_tqdm import weighted_kronbinations_tqdm, weighted_tqdm
from .Kron_Array import *

class kronbinations():
    # A class for scanning parameter landscapes
    # creates iterators over multiple parameters via kronecker products (itertools product), 
    # automatizes the construction of arrays to sotore results on the landscape, simplifies indexing 
    # while keeping the ability to add functions that get only executed in a specific subloop by tracking when a variable is changed

    def __init__(self, *values, **kwargs):
        # If values is a dictionary, also store the keys, so that change can be outputted by key
        weights_given = True if 'weights' in kwargs.keys() else False
        if isinstance(values[0], dict):
            if len(values) > 1:
                raise ValueError('If values is a dictionary, it must be the only argument')
            input_values = values[0]
            keys = input_values.keys()
            values = input_values.values()
            self.array_vars_all_names, self.array_vars_all, self.all_weights = [], [], []
            for i, (key, value) in enumerate(zip(keys, values)):  
                # check if value is a dict with value and weight
                self.array_vars_all_names.append(key)
                if isinstance(value, dict):
                    if 'value' in value.keys():
                        self.array_vars_all.append(value['value'])
                    else:
                        raise ValueError('If value is a dictionary, it must have a value key') 
                    if 'weight' in value.keys():
                        self.all_weights.append(value['weight'])
                    else:
                        self.all_weights.append(None)
                else: # Normal 
                    self.array_vars_all.append(value)
                    if weights_given:
                        self.all_weights.append(kwargs['weights'][i])
                    else:
                        self.all_weights.append(None)
            self.return_as_dict = True
        else:
            self.array_vars_all = list(values)
            if weights_given:
                self.all_weights = list(kwargs['weights'])
            else:
                self.all_weights = [None] * len(self.array_vars_all)
            self.return_as_dict = False
            self.array_vars_all_names = None
        for i, arr in enumerate(self.array_vars_all):
            # if does not have length, transform into array
            if not hasattr(arr, '__len__'):
                self.array_vars_all[i] = np.array([arr])
        # only relevant values in array_directions
        self.array_vars = [arr for arr in self.array_vars_all if len(arr) > 1]
        if isinstance(values, dict):
            self.array_vars_names = [name for name, arr in zip(self.array_vars_all_names, self.array_vars_all) if len(arr) > 1]
        # add index values of the array vars 
        self.array_vars_indexes = [i for i, arr in enumerate(self.array_vars_all) if len(arr) > 1] 
        self.weights = [w for i, w in enumerate(self.all_weights) if i in self.array_vars_indexes]

        if self.return_as_dict:
            self.curr_vals = {key: arr[0] for key, arr in zip(self.array_vars_all_names, self.array_vars_all)}
        else:
            self.curr_vals = [arr[0] for arr in self.array_vars_all]

        self.array_lengths_all = [len(arr) for arr in self.array_vars_all]
        self.array_lengths = [len(arr) for arr in self.array_vars]
        self.shape_all = tuple(self.array_lengths_all)
        self.shape = tuple(self.array_lengths)
        self.ndim_all = len(self.array_lengths_all)
        self.ndim = len(self.array_lengths)
        self.total_length = np.prod(self.array_lengths)
        self.size_all = np.prod(self.array_lengths_all)
        self.size = np.prod(self.array_lengths)
        
        self.index_list = [np.arange(len(v)) for v in self.array_vars]
        self.index_list_all = [np.arange(len(v)) for v in self.array_vars_all]
        # Define the iterators
        self.setup_iterator()

        self.do_index = True
        self.do_change = True
        self.do_tqdm = True
        self.set(**kwargs)   # redo these values if passed as kwargs
        # Initialize tqdm object
        self.pbar = weighted_kronbinations_tqdm(self.array_vars, self.weights, self.size)


    def empty(self, *var, **args):
        return Kron_Array(self.array_lengths, 'empty', *var, **args)
    def ones(self, *var, **args):
        return Kron_Array(self.array_lengths, 'ones', *var, **args)
    def zeros(self, *var, **args):
        return Kron_Array(self.array_lengths, 'zeros', *var, **args)
    def full(self, *var, **args):
        return Kron_Array(self.array_lengths, 'full', *var, **args)
    def random(self, *var, **args):
        return Kron_Array(self.array_lengths, 'random', *var, **args)

    def set(self, **args):
        key_substitution_list = [['index', 'do_index'], ['change', 'do_change'], ['progress', 'do_tqdm']]
        key_list = [v[0] for v in key_substitution_list]
        subs_list = [v[1] for v in key_substitution_list]
        for key, value in args.items():
            # Substitute certain keys from substitution list
            if key in key_list:
                key = subs_list[key_list.index(key)]
            if (key == 'return_as_dict' and value==True) and not isinstance(self.array_vars_all_names, list):
                raise ValueError('Keys are not defined, must create Object via dictionary in order to set "return_as_dict = True".')
            else:
                setattr(self, key, value)
        if self.do_tqdm:
            self.pbar = weighted_kronbinations_tqdm(self.array_vars, self.weights, self.size)
    def get(self, *args):
        key_substitution_list = [['index', 'do_index'], ['change', 'do_change'], ['progress', 'do_tqdm']]
        key_list = [v[0] for v in key_substitution_list]
        subs_list = [v[1] for v in key_substitution_list]
        x = []
        for key in args:
            if key in key_list:
                key = subs_list[key_list.index(key)]
            x.append(getattr(self, key))
        if len(x) == 1:
            return x[0]
        else:
            return x

    def __getitem__(self, key):
        # If the key is not in the data, return None
        if key in key_list:
            key = subs_list[key_list.index(key)]
            print(key)
        if key not in self.data:
            return None
        else:
            return self.data[key]

    def setup_iterator(self):
        self.product = itertools.product(*self.array_vars_all)
        self.indexes = itertools.product(*self.index_list)
        self.indexes_all = itertools.product(*self.index_list_all)

        last_indexes = -np.ones(self.ndim, dtype=int)
        last_indexes_all = -np.ones(self.ndim_all, dtype=int)
        last_values = [v[0] for v in self.array_vars_all]
        changed_var = np.zeros(self.ndim_all, dtype=bool)
        if self.return_as_dict:
            self.last_values = dict(zip(self.array_vars_all_names, last_values))
            self.last_indexes = last_indexes
            self.last_indexes_all = last_indexes_all
            self.changed_var = dict(zip(self.array_vars_all_names, changed_var))
        else:   
            self.last_values = last_values
            self.last_indexes = last_indexes
            self.last_indexes_all = last_indexes_all
            self.changed_var = changed_var

    def __next__(self):
        last_values = next(self.product)
        curr_index = next(self.indexes)
        curr_index_all = next(self.indexes_all)
        changed_var = tuple(np.not_equal(curr_index_all, self.last_indexes_all))
        if self.return_as_dict:
            self.last_values = dict(zip(self.array_vars_all_names, last_values))
            self.last_indexes = curr_index
            self.last_indexes_all = curr_index_all
            self.changed_var = dict(zip(self.array_vars_all_names, changed_var))
        else:   
            self.last_values = last_values
            self.last_indexes = curr_index
            self.last_indexes_all = curr_index_all
            self.changed_var = changed_var
        
        return self.last_values, self.last_indexes, self.changed_var

    def kronprod(self, **args):
        self.set(**args)
        if self.do_tqdm:
            # get all indexes
            self.pbar.init(np.array(list(itertools.product(*self.index_list))))
        if self.do_index:
            if self.do_change:
                for n in range(self.total_length):
                    v,i,c = next(self)
                    yield i, v, c
                    if self.do_tqdm:
                        self.pbar.increment()
            else:
                for n in range(self.total_length):
                    v,i,_ = next(self)
                    yield i, v
                    if self.do_tqdm:
                        self.pbar.increment()
        else:
            if self.do_change: 
                for n in range(self.total_length):
                    v,_,c = next(self)
                    yield v, c
                    if self.do_tqdm:
                        self.pbar.increment()
            else:  
                for n in range(self.total_length):
                    v,_,_ = next(self)
                    yield v
                    if self.do_tqdm:
                        self.pbar.increment()
        if self.do_tqdm:
            self.pbar.close()
        self.setup_iterator()
        
    def tqdm(self, iterator, weights=None, name='', **kwargs):
        if self.do_tqdm:
            return self.pbar.sub_tqdm(iterator, weights=weights, name=name, **kwargs)
        else:
            return weighted_tqdm(iterator, weights=weights, name=name, **kwargs)

    def changed(self, elem=None):
        if elem is None:
            return self.changed_var
        elif isinstance(elem, int):
            if self.return_as_dict:
                string = self.array_vars_all_names[elem]
                return self.changed_var[string]
            else:
                return self.changed_var[elem]
        elif isinstance(elem, str): # Outputs changed by key
            if isinstance(self.array_vars_all_names, list):
                return self.changed_var[elem]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')
                
    def index(self, elem=None):
        if elem is None:
            return self.last_indexes
        elif isinstance(elem, int):
            return self.last_indexes[elem]
        elif isinstance(elem, str): # By key
            if isinstance(self.array_vars_all_names, list):
                ind = self.array_vars_all_names.index(elem)
                return self.last_indexes_all[ind]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')

    def value(self, elem=None):
        if elem is None:
            return self.last_values
        elif isinstance(elem, int):
            # is dictionary?
            if self.return_as_dict:
                string = self.array_vars_all_names[elem]
                return self.last_values[string]
            else:
                return self.last_values[elem]
        elif isinstance(elem, str):
            if isinstance(self.array_vars_all_names, list):
                return self.last_values[elem]
            else:
                raise ValueError('Keys are not defined, must create Object via dictionary for this functionality.')
    
    def output_definition_kronprod(self, **args):
        self.set(**args)
        i = 'index'
        v = 'value'
        c = 'change'
        if self.do_index:
            if self.do_change:
                return i, v, c
            else:
                return i, v
        else:
            if self.do_change: 
                return v, c
            else:
                return v, 

    def all_combinations_array(self):
        # Generate an array for every combination of the input arrays
        # initialize arrays
        array_vars_all = []
        for i in range(self.ndim_all):
            curr_param = self.array_vars_all[i] # if curr_param doesn't have length, then make it an array
            if not hasattr(curr_param, '__len__'):
                curr_param = np.array([curr_param])
            curr_arr = np.empty(self.array_lengths_all, dtype=curr_param.dtype)
            # loop over all combinations
            for j in range(len(curr_param)):
                curr_arr[(slice(None), ) * i + (j,) ] = curr_param[j]
            array_vars_all.append(curr_arr)
        return array_vars_all
    
    