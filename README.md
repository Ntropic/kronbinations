# kronbinations
[![Python package](https://github.com/Ntropic/kronbinations/actions/workflows/python-package.yml/badge.svg)](https://github.com/Ntropic/kronbinations/actions/workflows/python-package.yml)  [![Coverage Status](https://coveralls.io/repos/github/Ntropic/kronbinations/badge.svg?branch=main)](https://coveralls.io/github/Ntropic/kronbinations?branch=main)

Install via 
`pip install kronbinations`

Import via
`from kronbinations import *`

## Description
kronbinations is used to remove nested loops, to perform multidimensional parameter sweeps and to generate arrays to store results of such sweeps.
### Usage: 
- Pass arbitrarily many arrays to the constructor `kronbinations`, to iterate over all combinations of the arrays elements.
```
k = kronbinations(array([1,2,3]), array(['a','b','c']), array([False,True,False]))
```
- If you need to store results of some computation in an array, you can construct these arrays via `k.empty()`, `k.ones()`, `k.zeros()`, , `k.full(fill_value)`, `k.randint()`, `k.rng_random(rng)` (and pass the corresponding numpy arguments , eg. `k.zeros(dtype=int)`).
- Finally you can iterate over all combinations using: 
```
for index, values, changed in k.kronprod(index=True, change=True, progress=True):
    # Demonstrating a few of the functions here
    if changed[0]:
        print('First value changed')
    x[index] = values[0]
```

## Authors: 
By Michael Schilling
