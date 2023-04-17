# -*- coding: utf-8 -*-
import numpy as np
import itertools
import warnings
import os


class Kron_Array(np.ndarray):   # Like JIT_Array but without saving and loading
    def __new__(cls, parent_shape, type, *args, **kwargs):
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
            shape_parent = parent_shape
            shape = tuple(list(shape_parent) +shape_added)
            if len(args) > 1:
                args = args[1:]
            else:
                args = ()
            added_shape = len(shape_added)
        else:
            shape = parent_shape
            shape_parent = parent_shape
            added_shape = 0
        if type=='ones':
            obj = np.ones(shape, *args, **kwargs).view(cls)
        elif type=='zeros':
            obj = np.zeros(shape, *args, **kwargs).view(cls)
        elif type=='full':
            obj = np.full(shape, *args, **kwargs).view(cls)
        elif type=='empty':
            obj = np.empty(shape, *args, **kwargs).view(cls)
        else:
            raise ValueError('type not recognized')
        obj.added_shape = added_shape # did we add to the shape of the parent
        obj.shape_parent = shape_parent
        return obj
    
    def arg_index_formatting(self, arg):
        if self.added_shape > 0: # check if tuple with first element tuple
            if isinstance(arg, tuple) and isinstance(arg[0], tuple):
                arg = arg[0] + arg[1:]
            if isinstance(arg, tuple) and isinstance(arg[0], np.ndarray): # add arg1 to arg0 copy elements of arg1 to arg0
                # transform ndarray of arg[0] into tuple and append arg[1]
                arg = tuple(arg[0].flatten()) + arg[1:]
        if not isinstance(arg, tuple):
            if isinstance(arg, int):
                arg = (arg,)
            elif isinstance(arg, list):
                arg = np.array(arg)
            elif arg is Ellipsis:
                arg = (arg,)
            elif isinstance(arg, slice):
                arg = (arg,)
            else:
               arg = tuple(arg)
        return arg

    def __getitem__(self, arg):
        # add zero to *args to make sure that the first index is not a tuple
        arg = self.arg_index_formatting(arg)
        return super().__getitem__(arg).view(np.ndarray)
    
    def __setitem__(self, arg, value):
        # add zero to *args to make sure that the first index is not a tuple
        arg = self.arg_index_formatting(arg)
        # set the values
        super().__setitem__(arg, value)

    def asarray(self):
        return np.asarray(self)
    def array(self):
        return self.asarray()
