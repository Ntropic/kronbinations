# -*- coding: utf-8 -*-
import numpy as np
from weighted_tqdm import weighted_kronbinations_tqdm, weighted_tqdm

import inspect
from hashlib import sha1
import os
import numpy as np
import inspect

# import JIT_Array and Kron_Fun_Modifier    
from .JIT_Array import *
from .Kron_Fun_Modifier import *

# An array class, that stores an array, and whether it's values have been calculated,
# and if so, what the values are
# If values of the array are requested, and they have not been calculated, they are calculated using a function handle passed to the constructor
class JIT_kronbinations():
    def __init__(self, *values, func=None, other_func=[], import_statements=[], other_arguments=[], checksum=None, checksum_pre='', autosave=True, data_dir='Cache', redo=False, progress=True, **kwargs):
        # Calculate checksums
        weights_given = True if 'weights' in kwargs.keys() else False
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_dir = data_dir
        if checksum is None:
            self.checksum = self.make_checksum(checksum_pre, func, *values, *import_statements, *other_arguments)
        else:
            self.checksum = checksum
        # check if data_dir exists
        
        self.autosave = autosave
        self.func = func
        self.other_func = other_func
        self.how_many_arrays_set = 0
        self.JIT_Arrays = []
        self.import_statements = import_statements#
        if isinstance(other_arguments, dict):
                self.other_arguments = [other_arguments]
        else:
            self.other_arguments = other_arguments

        # lengths of the values -> ignore the length one arrays, they are not to be iterated over
        # if values is a dictionary, then the keys are the directions, and the values are the arrays
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
        self.size_all = np.prod(self.array_lengths_all)
        self.size = np.prod(self.array_lengths)

        self.do_index = True
        self.do_change = True
        self.do_tqdm = progress
        self.redo = redo
        self.set(**kwargs)   # redo these values if passed as kwargs
        self.pbar = weighted_kronbinations_tqdm(self.array_vars, self.weights, self.size)
        
        self.curr_index = -1
        self.func_modifier()

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

    def change_all_dtype(self, object, dtype='int'):
        # navigates through the objects lists tuples and dictionaries until it finds a numpy array 
        # and changes the dtype if it is an integer to 'int'
        if isinstance(object, (list, tuple, dict)):
            # recursively call the function on the elements of the object
            is_tuple = isinstance(object, tuple)
            if is_tuple:
                object = list(object)
            if isinstance(object, dict):
                # navigate every key value pair
                for key, value in object.items():
                    object[key] = self.change_all_dtype(value, dtype=dtype)
            else: # isinstance(object, list):
                for i in range(len(object)):
                    object[i] = self.change_all_dtype(object[i], dtype=dtype)
            if is_tuple:
                object = tuple(object)
        elif isinstance(object, np.ndarray):
            if 'int' in str(object.dtype):
                object = object.astype(dtype)
        return object
    
    def make_checksum(self, checksum_pre, func, *args):
        # check every arg in args, if arg is an object of a class 
        # change all dtype int
        args = self.change_all_dtype(args)
        checksum_sha1 = sha1(str(args).encode('utf-8')).hexdigest()
        checksum_sin_fun = checksum_pre + checksum_sha1
        # get function name from func
        if func is not None:
            func_name = func.__name__
            checksum = checksum_pre + checksum_sha1 + func_name
            # check if files with checksum exist
            checksum_exists = False
            dir_list = os.listdir(self.data_dir)
            for file in dir_list:
                if checksum in file:
                    checksum_exists = True
                    break
            if not checksum_exists:
                # check if files with checksum_sin_fun exists -> rename to checksum
                for file in dir_list:
                    if checksum_sin_fun in file:
                        new_filename = file.replace(checksum_sin_fun, checksum)
                        os.rename(os.path.join(self.data_dir, file), os.path.join(self.data_dir, new_filename))  
                        print('  -> Renamed file from ' + file + ' to ' + new_filename)
        else:
            checksum = checksum_sin_fun
        return checksum

    def __getitem__(self, key):
        # If the key is not in the data, return None
        if key in key_list:
            key = subs_list[key_list.index(key)]
        if key not in self.data:
            return None
        else:
            return self.data[key]

    def save(self):
        # Save the data onto every JIT_Array
        for arr in self.JIT_Arrays:
            arr.save()

    # Replace these by LazyArrays and increase self.how_many_arrays_set by one every time
    def empty(self, *var, **args):
        return self.make_array('empty', *var, redo=self.redo, **args)
    def ones(self, *var, **args):
        return self.make_array('ones', *var, redo=self.redo, **args)
    def zeros(self, *var, **args):
        return self.make_array('zeros', *var, redo=self.redo, **args)
    def full(self, *var, **args):
        return self.make_array('full', *var, redo=self.redo, **args)

    def make_array(self, type, *var, redo=False, **args):
        # if name in **args, then use that as the name of the array and remove it from **args
        if 'name' in args:
            name = args['name']
            del args['name']
        else:
            name = None
        new_array = JIT_Array(self, type, *var, **args, name=name, redo=redo)
        self.JIT_Arrays.append(new_array)
        self.how_many_arrays_set += 1
        return new_array

    # This can be impolemented more eleganty by using a kronbinations like approach and defining func later
    def calculate(self, indexes):
        # indexes is a 2d array, where the first dimension is the index of the array, and the second dimension is the index of the value
        n = len(indexes)
        self.setup_iterator(indexes)
        # run function func with input arguments self, the JIT_arrays
        if not len(self.other_arguments)==0:
            _ = self.func(self, *self.other_arguments, *self.JIT_Arrays)
        else:
            _ = self.func(self,*self.JIT_Arrays)
        if self.autosave:
            self.save()

    def calculate_all(self):
        any_done = False
        # find where the JIT_Arrays are not done
        is_not_done = ~self.JIT_Arrays[0].calculated
        # find indexes where the JIT_Arrays are not done
        ind = np.where(is_not_done)
        # construct array from indexes, by concatenation
        indexes = np.array(ind).T
        #indexes = np.empty((len(ind[0]), len(ind)), dtype=int)
        #for i, index in enumerate(ind):
        #indexes[:, i] = index
        self.calculate(indexes)

    def construct_vals_and_all_indexes(self, index):
        # first construct self.indexes_all, then vals
        ind = np.zeros(self.ndim_all, dtype=int)
        vals = []
        ind[self.array_vars_indexes] = index
        for i in range(self.ndim_all):
            vals.append(self.array_vars_all[i][ind[i]])
        self.indexes_all = ind
        self.curr_vals = vals

    def setup_iterator(self, indexes):
        self.indexes = indexes
        self.total_length = len(indexes)
        last_indexes = tuple(-np.ones(self.ndim, dtype=int))
        last_indexes_all = tuple(-np.ones(self.ndim_all, dtype=int))
        changed_var = np.ones(self.ndim_all, dtype=bool)
        if self.return_as_dict:
            self.last_indexes = last_indexes
            self.last_indexes_all = dict(zip(self.array_vars_all_names, last_indexes_all))
            self.changed_var = dict(zip(self.array_vars_all_names, changed_var))
        else:   
            self.last_indexes = last_indexes
            self.last_indexes_all = last_indexes_all
            self.changed_var = changed_var
        self.curr_index = -1

    def __next__(self):
        self.curr_index += 1
        curr_index = tuple(self.indexes[self.curr_index])
        # construct current directions
        self.construct_vals_and_all_indexes(curr_index)
        last_values = self.curr_vals
        changed_var = tuple(np.not_equal(self.indexes_all, self.last_indexes_all))
        if self.return_as_dict:
            self.last_values = dict(zip(self.array_vars_all_names, last_values))
            self.last_indexes = curr_index #dict(zip(self.array_vars_all_names, curr_index))
            self.last_indexes_all = self.indexes_all
            self.changed_var = dict(zip(self.array_vars_all_names, changed_var))
        else:   
            self.last_values = last_values
            self.last_indexes = curr_index
            self.last_indexes_all = self.indexes_all
            self.changed_var = changed_var
        return self.last_values, self.last_indexes, self.changed_var

    def kronprod(self, **args):
        self.set(**args)
        if self.do_tqdm and self.total_length > 1:
            self.pbar.init(self.indexes)
        if self.do_index:
            if self.do_change:
                for n in range(self.total_length):
                    v,i,c = next(self)
                    yield tuple(i), v, c
                    if self.do_tqdm and self.total_length > 1:
                        self.pbar.increment()
            else:
                for n in range(self.total_length):
                    v,i,_ = next(self)
                    yield tuple(i), v
                    if self.do_tqdm and self.total_length > 1:
                        self.pbar.increment()
        else:
            if self.do_change: 
                for n in range(self.total_length):
                    v,_,c = next(self)
                    yield v, c
                    if self.do_tqdm and self.total_length > 1:
                        self.pbar.increment()
            else:
                for n in range(self.total_length):
                    v,_,_ = next(self)
                    yield v
                    if self.do_tqdm and self.total_length > 1:
                        self.pbar.increment()
        if self.do_tqdm and self.total_length > 1:
            self.pbar.close()
            
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

    # IF rng in func, then use Kron_Fun_Modifier to modify func and save it
    def func_modifier(self):
        func = self.func
        func_str = inspect.signature(func).parameters
        if 'rng' in func_str:
            data_dir = self.data_dir
            import_statements = self.import_statements
            fun_modifier = Kron_Fun_Modifier(func, self, data_dir, import_statements, other_func=self.other_func)
            func = fun_modifier.import_functions_from_file()
        self.func = func

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
