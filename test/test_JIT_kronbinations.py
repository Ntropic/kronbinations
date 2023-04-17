import pytest
import numpy as np
from kronbinations import JIT_kronbinations

# Generate lists of different dtypes
# and test the functions on them
floats = np.array([1.2, 3.4, 5.6, 7.8, 9.0])
ints = np.array([1, 3, 5, 7, 9], dtype=int)
strings = np.array(['a', 'b', 'c', 'd', 'e'])

# Create a kronbinations object
def test_kronbinations_matrices():
    # Create a kronbinations object
    # with the lists as values
    def fun(k, A, B):
        for i, v in k.kronprod(change=False, redo=False):
            A[i] = v[0]+v[1]+i[2]
            B[i] = v[0]-v[1]
        return A, B
    k = JIT_kronbinations(floats, ints, strings, func=fun, redo=False)
    assert k.size == 125
    assert k.shape == (5,5,5)
    assert k.ndim == 3
    # Test the generation of matrices
    o = k.ones(dtype=int)
    z = k.zeros(dtype=int)
    e = k.empty()
    f = k.full(fill_value= 1.0/3.0)
    assert o.shape == (5,5,5)
    assert z.shape == (5,5,5)
    assert e.shape == (5,5,5)
    assert f.shape == (5,5,5)
    
def test_kronbinations_loop_outputs():
    b = [False, True]
    # Check number of outputs
    for index in b:
        for change in b:
            how_many = 1 if index else 0
            how_many += 1 if change else 0
            if how_many == 0:
                how_many = 2 # elements of the values tuple - 1
                def funny(k, A):
                    for output in k.kronprod(index=index, change=change, progress=False):
                        assert len(output) == how_many+1
                k = JIT_kronbinations(floats, ints, strings, func=funny, redo=False, autosave=False)
                A = k.zeros()
                k.calculate_all()
                
def test_krombinations_intermediate_outputs():
    # Check the intermediate outputs
    def funny(k, A):
        for i,v,c in k.kronprod(index=True, change=True, progress=False):
            assert k.changed() == c
            assert k.index() == i # works since switching to tuples
            assert k.value() == v
            for j in range(3):
                assert k.changed(j) == c[j]
                assert k.index(j) == i[j]
                assert k.value(j) == v[j]
    k = JIT_kronbinations(floats, ints, strings, func=funny, redo=False, autosave=False)
    k.zeros()
    k.calculate_all()

def test_krombinations_settings_and_changes_to_them():
    def fun(k, A, B):
        for i, v in k.kronprod(change=False):
            A[i] = v[0]+v[1]+i[2]
            B[i] = v[0]-v[1]
        return A, B
    k = JIT_kronbinations(floats, ints, strings, func=fun, redo=False, autosave=False)
    k.set(do_index=True, do_change=True, do_tqdm=True)
    do_index, do_change, do_tqdm, return_as_dict = k.get('do_index', 'do_change', 'do_tqdm', 'return_as_dict')
    assert do_index
    assert do_change
    assert do_tqdm
    k.set(do_index=False, do_change=False, do_tqdm=False)
    do_index, do_change, do_tqdm, return_as_dict = k.get('index', 'do_change', 'do_tqdm', 'return_as_dict')
    assert not do_index
    assert not do_change
    assert not do_tqdm
    
def test_kronbinations_illegal_dict_outputs():
    # Check the illegal dictionary outputs
    def funny(k, A):
        for i,v,c in k.kronprod(index=True, change=True, progress=False):
            k.index('floats')
    k = JIT_kronbinations(floats, ints, strings, func=funny, redo=False, autosave=False)
    k.empty()
    
    def funny2(k, A):
        for i,v,c in k.kronprod(index=True, change=True, progress=False):
            k.value('floats')
    k2 = JIT_kronbinations(floats, ints, strings, func=funny2, redo=False, autosave=False)
    k2.empty()
    
    def funny3(k, A):
        for i,v,c in k.kronprod(index=True, change=True, progress=False):
            k.changed('floats')
    k3 = JIT_kronbinations(floats, ints, strings, func=funny3, redo=False, autosave=False)
    k3.empty()
    
    with pytest.raises(ValueError):
        k.calculate_all()
    with pytest.raises(ValueError):
        k2.calculate_all()
    with pytest.raises(ValueError):
        k3.calculate_all()

def test_kronbinations_dict_objects():    
    # Create a kronbinations object
    # with the lists as a dictionary
    keys = ['floats', 'ints', 'strings']
    d = {'floats': floats, 'ints': ints, 'strings': strings}
    def funny(k, A):
        for i,v,c in k.kronprod(index=True, change=True, progress=False):
            for j, key in enumerate(keys):
                assert k.changed(key) == c[key]
                assert k.index(key) == i[j]
                assert k.value(key) == v[key]
    with pytest.raises(ValueError):
        k = JIT_kronbinations(d, d, func=funny, redo=False, autosave=False)
    k = JIT_kronbinations(d, func=funny, redo=False, autosave=False)
    k.empty()
    k.calculate_all()