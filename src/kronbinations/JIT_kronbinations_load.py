# -*- coding: utf-8 -*-
import numpy as np
from weighted_tqdm import weighted_kronbinations_tqdm, weighted_tqdm

from hashlib import sha1
import os
import numpy as np


# An array class, that stores an array, and whether it's values have been calculated,
# and if so, what the values are
# If values of the array are requested, and they have not been calculated, they are calculated using a function handle passed to the constructor
class JIT_kronbinations_load():
    def __init__(self, *values, func=None, import_statements=[], other_arguments=[], checksum=None, checksum_pre='', data_dir='Cache', **kwargs):
        # Calculate checksums
        self.data_dir = data_dir
        # check if data_dir exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if checksum is None:
            self.checksum = self.make_checksum(checksum_pre, func, *values, *import_statements, *other_arguments)
        else:
            self.checksum = checksum
        # Check if files with checksum exist
        filenames, var_names = self.filenames()
        n_filenames = len(filenames)
        if n_filenames == 0:
            raise ValueError('No files with checksum ' + self.checksum + ' found!')
        # Load files
        self.vars = {}
        self.var_calculated = {}
        for i, (filename, var_name) in enumerate(zip(filenames, var_names)):
            new_var, new_calculated, _ = self.load_filename(filename)
            self.vars[var_name] = new_var
            self.var_calculated[var_name] = new_calculated

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
                else: # Normal 
                    self.array_vars_all.append(value)
            self.return_as_dict = True
        else:
            self.array_vars_all = list(values)
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
        self.set(**kwargs)   # redo these values if passed as kwargs
        
    def get_arrays(self):
        return self.vars
    
    def get_calculated(self):
        return self.var_calculated
    
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
        if key not in self.data:
            return None
        else:
            return self.data[key]

    def load_filename(self, filename):
        try:
            with np.load(os.path.join(self.data_dir, filename), allow_pickle=True) as f:
                var = f['var']
                calculated = f['calculated']
                how_many_missing = f['how_many_missing']
        except:
            raise Exception('Error loading file: ' + filename)
        return var, calculated, how_many_missing
            
    def filenames(self):
        # searches for filenames in data_dir with checksum and ending on .npz
        filenames = []
        var_names = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.npz') and file.startswith(self.checksum):
                filenames.append(file)
                new_file = file.replace(self.checksum + '_', '')
                new_file = new_file.replace('.npz', '')
                var_names.append(new_file)
        return filenames, var_names
    
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
        else:
            checksum = checksum_sin_fun
        return checksum
