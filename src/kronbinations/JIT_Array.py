# -*- coding: utf-8 -*-
import numpy as np
import itertools
import warnings
import os


class JIT_Array(np.ndarray):
    def __new__(cls, parent, type, *args, redo=False, name=None, **kwargs):
        data_dir = parent.data_dir
        checksum = parent.checksum
        if name is None:
            name = type + str(parent.how_many_arrays_set)
        filename = os.path.join(data_dir, checksum + '_' + name + '.npy')
        file_exists = os.path.isfile(filename)
        if file_exists and not redo:
            try:
                var, done, how_many_missing = np.load(filename, allow_pickle=True)
            except OSError:
                print('Error: Loading file. ' +  filename)
            obj = var.view(cls)
            obj.calculated = done
            obj.how_many_missing = how_many_missing
            shape_parent = parent.shape
            if len(obj.shape) > len(shape_parent):
                added_shape = len(obj.shape) - len(shape_parent)
                shape_added = list(obj.shape[-added_shape:])
            else:
                added_shape = 0
                shape_added = []
        else:
            # first args is an addition to shape, so we need to add it to the shape of the parent 
            if len(args) > 0: #parent.shape is a tuple
                if isinstance(args[0], tuple):
                    shape_added = list(args[0])
                elif isinstance(args[0], int):
                    shape_added = [args[0]]
                elif isinstance(args[0], list):
                    shape_added = args[0]
                else:
                    shape_added = []
                shape_parent = parent.shape
                shape = tuple(list(shape_parent) +shape_added)
                if len(args) > 1:
                    args = args[1:]
                else:
                    args = ()
                added_shape = len(shape_added)
            else:
                shape = parent.shape
                shape_parent = shape
                added_shape = 0
            if type=='ones':
                obj = np.ones(shape, *args, **kwargs).view(cls)
            elif type=='zeros':
                obj = np.zeros(shape, *args, **kwargs).view(cls)
            elif type=='full':
                obj = np.full(shape, *args, **kwargs).view(cls)
            else:
                obj = np.empty(shape, *args, **kwargs).view(cls)
            obj.calculated = np.zeros(shape_parent, dtype=bool)
            obj.how_many_missing = np.prod(shape_parent)  
        obj.added_shape = added_shape # did we add to the shape of the parent
        obj.shape_parent = shape_parent
        obj.ndim_parent = len(shape_parent)
        obj.parent = parent
        obj.name = name
        obj.checksum = checksum
        obj.data_dir = data_dir
        obj.filename = filename
        obj.file_exists = file_exists
        return obj
    
    # A function to save the matrix
    def save(self):
        # if directory doesn't exist, create it
        filename = self.filename
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
            except OSError:
                print('Error: Creating directory. ' +  self.data_dir)
        try:
            # ignore Warnings for saving
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                np.save(filename, (np.asarray(self), np.asarray(self.calculated), self.how_many_missing), allow_pickle=True)
        except OSError:
            print('Error: Saving file. ' +  filename)
        self.file_exists = True

    def slice_to_tuple(self, s, array_shape):
        if isinstance(s, slice):
            return np.array(range(*s.indices(array_shape)))
        else:
            # return as iterable
            if isinstance(s, (int, np.int32, np.int64)):
                return np.array([s])
            elif isinstance(s, list):
                return np.array(s)
            else:
                return s
    # function that transforms a tuple of slices into index combinations
    # Construct all combinations of indexes from slices
    def slice_tuple_to_tuple(self, slice_tuple, array_shape):
        # check for ellipsis and add slices if needed
        if slice_tuple[0] is Ellipsis:
            # remove ellipsis and add slices to the beginning of the tuple
            slice_tuple = (slice(None, None, None),)*(len(array_shape)-len(slice_tuple)+1) + slice_tuple[1:]
        elif slice_tuple[-1] is Ellipsis:
            # remove ellipsis and add slices to the end of the tuple
            slice_tuple = slice_tuple[:-1] + (slice(None, None, None),)*(len(array_shape)-len(slice_tuple)+1)
        else:
            if len(slice_tuple) < len(array_shape):
                slice_tuple = slice_tuple + (slice(None, None, None),)*(len(array_shape)-len(slice_tuple))
        return [self.slice_to_tuple(s, a) for s, a in zip(slice_tuple, array_shape)]

    def find_element_indexes(self, slice_tuple, array_shape, indexes):
        tuples = self.slice_tuple_to_tuple(slice_tuple, array_shape)
        lengths = [1]
        lengths += [len(t) for t in np.flip(tuples[1:])]
        len_prod = np.flip(np.cumprod(lengths))
        len_tuples = len(tuples)
        # find indexes in the tuples using mod operations
        inds = np.zeros([len(indexes), len_tuples], dtype=int)
        for i in range(len_tuples):
            inds[:,i] = tuples[i][ indexes // len_prod[i] ]
            indexes = indexes % len_prod[i]
        return inds
    
    def find_all_elements(self, slice_tuple, array_shape):
        tuples = self.slice_tuple_to_tuple(slice_tuple, array_shape)
        lengths = [1]
        lengths += [len(t) for t in np.flip(tuples[1:])]
        how_many = np.prod(lengths)
        inds = np.zeros([how_many, len(lengths)], dtype=int)
        # use itertools combinations to find all combinations of indexes
        for i, t in enumerate(itertools.product(*tuples)):
            inds[i,:] = t
        return inds
    
    def arg_index_formatting(self, arg):
        #print('arg_index_formatting', arg)
        if self.added_shape > 0: # check if tuple with first element tuple
            if isinstance(arg, tuple) and isinstance(arg[0], tuple):
                arg = arg[0] + arg[1:]
            #if isinstance(arg, tuple) and isinstance(arg[0], np.ndarray): # add arg1 to arg0 copy elements of arg1 to arg0
            #    # transform ndarray of arg[0] into tuple and append arg[1]
            #    arg = tuple(arg[0].flatten()) + arg[1:]
        find_indexes = True
        if not isinstance(arg, tuple):
            if isinstance(arg, int):
                arg = (arg,)
            elif isinstance(arg, list):
                arg = np.array(arg)
                find_indexes = False
            elif isinstance(arg, np.ndarray):
                find_indexes = False
            if arg is Ellipsis:
                arg = (arg,)
                find_indexes = True
            elif isinstance(arg, slice):
                arg = (arg,)
                find_indexes = True
            else:
               arg = (arg,)
        return arg, find_indexes

    def __getitem__(self, arg):
        # add zero to *args to make sure that the first index is not a tuple
        arg, find_indexes = self.arg_index_formatting(arg)
        arg_shortened = arg[:self.ndim_parent]
        if self.how_many_missing > 0:
            items = self.calculated.__getitem__(arg_shortened).flatten() # 
            if not np.all(items):
                # Transform 
                # find the indices that are not yet calculated and calculate them, outputs a list of index combinations
                calc_indexes = np.where(~items)[0]
                if find_indexes == True:
                    indexes = self.find_element_indexes(arg_shortened, self.shape_parent, calc_indexes)
                else:
                    indexes = arg_shortened
                self.parent.calculate(indexes)
        return super().__getitem__(arg).view(np.ndarray)
    
    def is_done(self, index):
        return self.calculated[index]

    def __setitem__(self, arg, value):
        # add zero to *args to make sure that the first index is not a tuple
        arg, _ = self.arg_index_formatting(arg)
        arg_shortened = arg[:self.ndim_parent]
        # get the corresponding calculated items
        if self.how_many_missing > 0:
            number_of_not_yet_set = np.sum(self.calculated[arg_shortened].flatten() == False)
            self.calculated.__setitem__(arg_shortened, True)
            self.how_many_missing -= number_of_not_yet_set
        # set the values
        super().__setitem__(arg, value)

    #def names2dirs(self, *names):
    #    # finds the directions of the hyperarray from the names, then returns the directions

    def asarray(self):
        if self.how_many_missing > 0:
            self.parent.calculate_all()
        return np.asarray(self)
    
    def array(self):
        return self.asarray()
    
    #def dirs_array(self, *names):
    #    # picks out the directions from the array specified by names, 