# kronbinations

Install via 
`pip install kronbinations`

Import via
`from kronbinations import *`

## Description
kronbinations is used to remove nested loops, to perform multidimensional parameter sweeps and to generate arrays to store results of such sweeps.
### Input: 
`n_s = (n_1, n_2, ..., n_m)` 
Tuple of integers (or array when using `mrange_array`)
### Output: 
Generator outputting `len(n_s)` values with every call, generating every combination of values in the intervals `[0,1,...,n_s[i]-1]`
### Use:
```
for a, b, ..., m in mrange((n_a, n_b, ..., n_m)):
    ...
Replaces:
for a in range(n_a):
    for b in range(n_b):
       ...
           for m in range(n_m):
               ...
```
## Authors: 
By Michael Schilling