import inspect
import importlib
import re # regular expressions
import os
import numpy as np

class Kron_Fun_Modifier:
    # Modifies function handles containing kronbinations and kronprod terms, modifies them for use with JIT_Arrays
    def __init__(self, func, parent, data_dir='Cache', import_statements=None, other_func=[]):
        self.func = func
        self.other_func = other_func
        self.import_statements = import_statements
        self.parent = parent
        self.data_dir = data_dir

        self.func_str = inspect.getsource(self.func)
        # separate semicolon separated statements into separated lines, and unify multi line statements inside of brackets
        self.func_str = self.simple_reformatting()
        # Separate function into def, body, name, input arguments and output arguments
        self.definition, self.body, self.name, self.input_args = self.separate()  # , self.return_args, self.number_of_return_statements
        self.indentation_depths, self.begin_indentation, self.end_indentation, self.indentation_types = self.classify_indentations()
        self.all_variables, self.first_appearance = self.find_all_variables()
        # Find variables by line
        # go line by line and find variables by line
        self.body_classification = {'variables': [], 'where': [], 'indexed': [], 'defined': []}
        for i, line in enumerate(self.body):
            variables, where, indexed, defined = self.variables_in_line(line)
            self.body_classification['variables'].append(variables)
            self.body_classification['where'].append(where)
            self.body_classification['indexed'].append(indexed)
            self.body_classification['defined'].append(defined)
        # separate body into intro and loop
        _, _, self.kronbinations_index = self.before_after()
        #do_rng = self.contains_rng()
        string = self.body[self.kronbinations_index]
        for_variables, for_variable_types = self.classify_for_variables_kronprod(string)
        self.new_body = self.rng_conditions_stack(for_variables, for_variable_types)
        #print(self.construct_function_string())
        self.file_name = self.save()

    def save(self):
        file_name = os.path.join(self.data_dir, self.name + '.py')
        with open(file_name, 'w') as f:
            f.write(self.construct_function_string())
        # return file_name
        return file_name

    def before_after(self, input_lines=None, separator='kronprod', *lists):
        # separates function into intro with variable definitions and loop with kronbination
        # detect kronprod as separator
        # detect kronbination as initiator
        if input_lines is None:
            input_lines = self.body
        n_lines = len(input_lines)
        line_index = -1
        curr_index = 0
        line_gen = iter(input_lines)
        while line_index < 0 and curr_index < n_lines:
            curr_line = next(line_gen)
            if separator in curr_line:
                line_index = curr_index
            curr_index += 1
        if line_index < 0:
            raise ValueError('Separator not found')
        before = input_lines[:line_index]
        after = input_lines[line_index:]
        # separate *lists into before and after
        if len(lists) > 0:
            before_lists = []
            after_lists = []
            for l in lists:
                before_lists.append(l[:line_index])
                after_lists.append(l[line_index:])
            return before, after, before_lists, after_lists
        return before, after, line_index
        

    def simple_reformatting(self):
        # reformat statements so that ; separated statements are on separate lines and multi line statements are on one line
        func_str = self.func_str
        # separate string into a list of strings by line   
        func_str_lines = func_str.splitlines()
        # remove empty lines
        func_str_lines = [line for line in func_str_lines if line.strip() != '']
        # remove comments
        new_func_str_lines = []
        for line in func_str_lines:
            if '#' in line:
                remaining_str = line.split('#')[0]
                if remaining_str.strip() != '':
                    new_func_str_lines.append(remaining_str)
            else:
                new_func_str_lines.append(line)
        func_str_lines = new_func_str_lines
        # remove docstrings
        new_func_str_lines = []
        currently_in_docstring = False
        for line in func_str_lines:
            if "'''" in line:
                # check if appears twice in the line
                if not line.count("'''") == 2:
                    currently_in_docstring = not currently_in_docstring   # Toggle
                    # keep start of the string
                    if currently_in_docstring:
                        # docstring is starting
                        #keep beginning of the string
                        beginning = line.split("'''")[0]
                        if beginning.strip() != '':
                            new_func_str_lines.append(beginning)
            elif not currently_in_docstring:
                new_func_str_lines.append(line)
        func_str_lines = new_func_str_lines
        # separate semicolon separated statements into separate lines
        new_func_str_lines = []
        for line in func_str_lines:
            if ';' in line:
                new_func_str_lines += line.split(';')
            else:
                new_func_str_lines.append(line)
        func_str_lines = new_func_str_lines

        # unify multi line statements inside of brackets, find open brackets that aren't closed
        new_func_str_lines = []
        currently_in_brackets = np.array([0,0,0])
        open_braces = ['(', '[', '{']
        close_braces = [')', ']', '}']
        last_in_brackets = False
        for line in func_str_lines:
            if last_in_brackets == False:
                curr_line = line.rstrip() 
            for i, brace in enumerate(open_braces):
                currently_in_brackets[i] += line.count(brace)
            for i, brace in enumerate(close_braces):
                currently_in_brackets[i] -= line.count(brace)
            if np.any(currently_in_brackets < 0):
                raise ValueError('Too many closing brackets')
            
            if np.sum(currently_in_brackets):
                if last_in_brackets == True:
                    curr_line += ' ' + line.strip()
                    last_in_brackets = True
            else:
                if last_in_brackets == True:
                    curr_line += ' ' + line.strip()
                new_func_str_lines.append(curr_line)
                last_in_brackets = False
            if np.any(currently_in_brackets > 0):
                last_in_brackets = True
        func_str_lines = new_func_str_lines
        # combine lines into one string
        func_str = ''
        for line in func_str_lines:
            func_str += line + '\n'
        return func_str
    
    # Separate function into def, body, name, input arguments and output arguments
    def separate(self):
        func_str = self.func_str
        # separate string into a list of strings by line
        func_str_lines = func_str.splitlines()
        definition = func_str_lines[0]
        body = func_str_lines[1:]
        ### name is the first word after def
        name = definition.split()[1].split('(')[0]
        ### input arguments
        input_args_unformatted = definition.split('(')[1].split(')')[0]
        # remove equal signs, the default values after equal signs and remove spaces
        input_args = [arg.split('=')[0].strip() for arg in input_args_unformatted.split(',')]
        return definition, body, name, input_args

    # get indentation depth of each line in the body, beginning of indentations, ends and types
    def indentation_blocks(self, indentations_by_line):
        """Return a list of tuples of the form (start_line, end_line) for each block of code."""
        # indentations_by_line is an array of the indentation level of each line (int)
        # ind start is an index to be added to the start of the block and the end of the block
        # do it iteratively
        start = []
        ender = []
        len_ind = len(indentations_by_line)
        curr_ind = -1
        for i in range(1, len_ind):
            if indentations_by_line[i] > indentations_by_line[i-1]:
                curr_indentations = indentations_by_line[i]
                start.append(i - 1)
                # find end of block
                for j in range(i+1, len_ind):
                    if indentations_by_line[j] < curr_indentations:
                        ender.append(j)
                        break
        return start, ender
    def classify_indentations(self, input_lines=None):
        if input_lines is None:
            input_lines = self.body
        indentation_depths = np.array([len(line) - len(line.lstrip()) for line in input_lines])
        lines = [line.strip() for line in self.body]
        tab_to_spaces = np.min(indentation_depths)
        #print(indentation_depths)
        indentation_depths = indentation_depths // tab_to_spaces
        begin_indentation, end_indentation = self.indentation_blocks(indentation_depths)
        indentation_types = []
        for i in range(len(begin_indentation)):
            curr_type = lines[begin_indentation[i]].strip().split()[0]
            if curr_type == 'else:':
                curr_type = 'else'
            indentation_types.append(curr_type)
        return indentation_depths, begin_indentation, end_indentation, indentation_types

        # function to detect an equal sign in a line
    def brace_depth(self, line):
        open_braces = ['(', '[', '{']
        closed_braces = [')', ']', '}']
        all_braces = open_braces + closed_braces
        # for every char get the brace level and store whether it is a brace
        n = len(line)
        brace_levels = - np.zeros(n, dtype=int)
        brace_types = - np.ones(n, dtype=int)
        last_brace = []
        curr_brace_level = 0
        for i, char in enumerate(line):
            # New brace? 
            if curr_brace_level < 0:
                print('Error: brace level is negative')
            if char in all_braces:
                if char in open_braces:
                    curr_brace_level += 1
                    last_brace.append(open_braces.index(char))
                    # don't store brace level and type during a brace
                elif char in closed_braces:
                    curr_brace_level -= 1
                    # delete last brace element
                    last_brace.pop()
                    # don't store brace level and type during a brace
                brace_levels[i] = -1
            else:
                brace_levels[i] = curr_brace_level
                if curr_brace_level > 0:
                    brace_types[i] = last_brace[-1]      
        return brace_levels, brace_types
    def detect_equal_sign(self, line):
        # ignore equal signs in braces
        if not '=' in line:
            return 0
        else:
            # index of first occurance
            index = line.find('=')
            prev_char = line[index-1]
            # occurance must also not be !=, ==, >=, <=, 
            list_of_equal_signs = ['!', '>', '<']
            if prev_char in list_of_equal_signs:
                return 0
            next_char = line[index+1]
            if next_char == '=':
                return 0
            # occurance must be at brace_level = 0
            brace_levels, brace_types = self.brace_depth(line[:index])
            if brace_levels[-1] > 0:
                return 0
            return index
    def detect_for_statement(self, line):
        # ignore equal signs in braces
        if not 'for ' in line:
            return False, [], []
        else:
            # index of first occurance
            index_for = line.find('for ')
            index_in = line[index_for:].find(' in ')+index_for
            # is there an in statement after the for statement
            if not 'in' in line[index_for:]:
                print("A ' for ' statement must be followed by an ' in ' statement: ", line)
                return False, [], []
            else:
                # find first index after for statement and left index before in statement for every for ... in ... statement in line
                left =  [index_for+3]
                right = [index_in]
                # find other occurances
                while 'for' in line[index_for+1:]:
                    index_for = line[index_for+1:].find('for ')+index_for+1
                    index_in = line[index_in+1:].find(' in ')+index_in+1
                    left.append(index_for+3)
                    right.append(index_in)
            return True, left, right
    # function to separate comma separated variables
    def separate_variables(self, variable):
        # separate by commas or braces
        # first remove braces ()
        if '(' in variable:
            # check if ) in variable
            if ')' in variable:
                # are there variables before   -> does not yet support partial tuple unpacking, eg. a, (b, c) = d
                # replace ( by ,
                variable = variable.strip()
                while variable[0] == '(':
                    variable = variable[1:]
                while variable[-1] == ')':
                    variable = variable[:-1]
                # remove the other braces
                variable = variable.replace('(','')
                variable = variable.replace(')', '')
                variable = variable.replace(' ','')
            else:
                don_t_add = True
                print('Warning: could not parse line: '+variable, ' (missing closing parenthesis)')
        # last thing, separate multiple variable assignment by commas
        if ',' in variable:
            variable = variable.split(',')
            variable = [var.strip() for var in variable]
        if isinstance(variable, str):
            variable = [variable]
        return variable
    # find all variables in function
    def find_all_variables(self):
        # find all variables in function (variables) and their first appearances (first_appearances)
        # get variables in input arguments
        variables = [arg for arg in self.input_args]
        first_appearances = [-1 for i in range(len(self.input_args))]
        # function to remove indexes from variables a[1] --> a
        def remove_indexes(variable):
            if '[' in variable:
                variable = variable.split('[')[0].strip()
            return variable
        def extract_variables(string):
            # , separated and remove ()
            # remove indexes
            vars = string.replace('(','').replace(')','').strip().split(',')
            return [remove_indexes(var.strip()) for var in vars]
        for i, l in enumerate(self.body):
            line = l.strip()
            # find equal signs
            equal_sign = self.detect_equal_sign(line)
            # find for statements
            for_statement, left, right = self.detect_for_statement(line)
            # find all variables in these statements
            if equal_sign:
                # get variables before
                curr_variables = extract_variables(line[:equal_sign])
                curr_appearances = [i for j in range(len(curr_variables))]
            elif for_statement:# only consider first for statement
                curr_variables = extract_variables(line[left[0]:right[0]])
                curr_appearances = [i for j in range(len(curr_variables))]
            else:
                curr_variables = []
                curr_appearances = []
            # add variables to list
            variables += curr_variables
            first_appearances += curr_appearances

        # make variables unique, but remove same elements in first_appearance
        #for var, first in zip(variables, first_appearances):
        #    print(var, first)
        variables, indexes = np.unique(variables, return_index=True)
        first_appearances = [first_appearances[i] for i in indexes]
        # sort appearances anmd get sorting indexes
        first_appearances, indexes = np.sort(first_appearances), np.argsort(first_appearances)
        variables = [variables[i] for i in indexes]
        return variables, first_appearances


    def variables_in_line(self, line):
        if isinstance(line, int):
            line = self.body[line]
        variable_list = self.all_variables
        # find all variables in line
        def remove_equal_sign(line):
            # a string sthat should be separated at an equal sign
            # relevant part is after equal sign remove everything before it
            return line[line.find('=')+1:].strip()
        def is_indexed(string, end_index):
            # checks if a variable is indexed and removes the index, check for  [
            n = len(string)
            curr_index = end_index    
            while curr_index < n:
                if string[curr_index] == ' ': # ignore spaces
                    curr_index += 1
                elif string[curr_index] == '[': # found index
                    return True
                else:
                    return False
        line = line.strip()
        # find equal signs
        equal_sign = self.detect_equal_sign(line)
        # find for statements
        for_statement, left, right = self.detect_for_statement(line)
        # brace levels and types
        brace_levels, brace_types = self.brace_depth(line)
        variables = []
        indexed = []
        defined = []    
        where = []
        # find all variables in these statements
        separators = ['=', '+', '-', '*', '/', '>', '<', ',', '.', '%', '&', '|', '^', '~', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', ' '] #symbols that can end or start a variable name
        for var in variable_list:
            # are there any occurances of var in line
            if var in line:
                # find all occurances of var in line
                indexes = [m.start() for m in re.finditer(var, line)]
                n = len(var)
                # check if var is indexed var[...
                curr_vars = []
                curr_where = []
                curr_indexed = []
                curr_defined = []
                for i in indexes:
                    # is this a true hit or just a substring of a longer function or variable name?   -> compare adjacent characters to separators
                    if i > 0: # to the left
                        if line[i-1] not in separators:
                            continue # skip this iteration
                    if i+n < len(line): # to the right
                        if line[i+n] not in separators:
                            continue
                    # if we get here, we have a true hit, still need to classify this variable
                    curr_vars.append(var)
                    curr_where.append(i)
                    curr_is_indexed = is_indexed(line, i+n)
                    curr_indexed.append(curr_is_indexed)
                    # is defined                
                    if curr_is_indexed:
                        curr_defined.append(False)
                    else:
                        # left of equal sign?
                        if equal_sign > i:
                            if brace_types[i] == 1:  # is within square brackets []
                                curr_defined.append(False)
                            else:
                                curr_defined.append(True)
                        elif for_statement: # is within for...in statement?
                            cdef = False
                            j = 0
                            while j <len(left) and cdef == False:
                                if left[j] < i and i < right[j]:
                                    cdef = True
                                j += 1
                            curr_defined.append(cdef)
                        else:
                            curr_defined.append(False)
                variables += curr_vars
                where += curr_where
                indexed += curr_indexed
                defined += curr_defined
        # sort by location in line (where) from right to left 
        indexes =np.argsort(where)[::-1]
        where = [where[i] for i in indexes]
        variables = [variables[i] for i in indexes]
        indexed = [indexed[i] for i in indexes]
        defined = [defined[i] for i in indexes]

        return variables, where, indexed, defined   
    
    def remove_from_definition(self, var):
        # remove the variable var from the definition (first line of function)
        definition = self.definition
        # find variables in all_variables, match first_appearance
        if var in self.all_variables:
            ind = self.all_variables.index(var)
            first = self.first_appearances[ind]
            if first >= 0: # is defined in function
                print('Variable not found in function definition')
                return definition
            # remove variable from definition
            index_left = definition.find('(')
            # find after index_left
            index_var = definition.find(var, index_left) 
            len_var = len(var)
            # remove from string
            new_definition = definition[:index_var] + definition[index_var+len_var:]
            return new_definition
        else:
            print('Variable not found in function definition')
            return definition

    # use rng?
    def contains_rng(self, line_indexes=None):  #line_indexes = [min, max] or simply index
        all_variables = self.all_variables
        first_appearances = self.first_appearance
        if not 'rng' in all_variables:
            return False
        else:
            if line_indexes is None:
                if first_appearances[all_variables.index('rng')] < 0:
                    return True
                else:
                    return False
            else:
                if isinstance(line_indexes, int):
                    variables, _, _, _ = self.variables_in_line(line_indexes)
                    if 'rng' in variables:
                        return True
                else:
                    for i in range(line_indexes[0], line_indexes[1]):
                        # variables in line
                        variables, _, _, _ = self.variables_in_line(i)
                        if 'rng' in variables:
                            return True
                return False
    
    def classify_for_variables_kronprod(self, string):
        def list_of_arguments(string):
            string=string.replace(' ', '') # remove spaces
            # separate string iinto a list of strings, and elements within braces into subslists of strings, so that a, (b, c) becomes ['a', ['b', 'c']]
            # separate string into a list of different brace depths, so that a, (b, c) becomes ['a', ['b, c']]
            
            # find braces first
            brace_depth = 0
            brace_depths = np.empty(len(string), dtype=int)
            for i, c in enumerate(string):
                if c == '(':
                    brace_depth += 1
                elif c == ')':
                    brace_depth -= 1
                brace_depths[i] = brace_depth
            # now split the string into a list of strings, each with the same brace depth
            curr_depth = 0
            curr_string = ''
            strings = []
            for i, c in enumerate(string):
                if brace_depths[i] == curr_depth:
                    curr_string += c
                elif brace_depths[i] < curr_depth:
                    strings.append( [curr_string] )
                    curr_string = ''
                    curr_depth = brace_depths[i]
                else: #if brace_depths[i] > curr_depth:
                    strings.append(curr_string)
                    curr_string = ''
                    curr_depth = brace_depths[i]
            else:
                strings.append(curr_string)
            # now split by commas, strip and remove empty strings
            def clean_str_list(str_list):
                new_str_list = []
                for s in str_list:
                    if isinstance(s, str):
                        s = s.strip()
                        if s != '':
                            s = s.replace(',', '')	
                            s_split = s.split(' ')
                            s_split = [s.strip() for s in s_split]
                            new_str_list +=s
                    else:
                        new_str_list.append(clean_str_list(s))
                return new_str_list
            return clean_str_list(strings)
        _, left, right = self.detect_for_statement(string)
        sub_string = string[left[0]:right[0]].strip()
        variables = list_of_arguments(sub_string)
        # get arguments to detect changed indexing, change, progress behaviour
        argument_string = string[right[0]:].strip()
        #find first '(' and last ')'
        left = argument_string.find('(')
        right = argument_string.rfind(')')
        argument_string = argument_string[left+1:right]
        # split by comma
        arguments = argument_string.split(',')
        arguments = [a.strip().split('=') for a in arguments]
        def str_2_bool(s):
            return True if s == 'True' else False
        argument_dict = {key.strip(): str_2_bool(value.strip()) for key, value in arguments}
        names = self.parent.output_definition_kronprod(**argument_dict)
        if not isinstance(names, tuple):
            names = (names, )
        for_variables = []
        for_variable_types = []
        if not len(names) == len(variables):
            raise ValueError('kronprod returns {} variables, but {} variables are defined in for loop'.format(len(names), len(variables)))
        for i, name in enumerate(names):
            for_variables += variables[i]
            for_variable_types += [name for var in variables[i]]
        return for_variables, for_variable_types
    #def replace_braces_outside_of_function_calls(self, string):
    def change_kronbinations_calls(self, string):
        # change k.changed() to k.value()
        # first find variable name of kronbinations_object in kronprod line
        kronbinations_index = self.kronbinations_index
        line = self.body[kronbinations_index]
        # find variable name
        var_name = line.split('.kronprod')[0].strip().split(' ')[-1]
        # find var_name.changed() and replace with var_name.value() 
        # find all instances of var_name.changed()
        changed = var_name + '.changed('
        value = var_name + '.value('
        index = var_name + '.index('
        all_three = [changed, value, index]
        # find all instances:
        var_list = []
        for i in range(len(all_three)):
            curr_var = all_three[i]
            # find all occurances of curr_var in string one after another
            new_ind = string.find(curr_var)
            ind_begin = []
            while new_ind >= 0:
                ind_begin.append(new_ind)
                new_ind = string.find(curr_var, new_ind+len(curr_var))
            #ind_begin = [m.start() for m in re.finditer(curr_var, string)]
            # find end of each instance
            ind_end = []
            for j in range(len(ind_begin)):
                ind = ind_begin[j]
                #check if the character before ind_begin is a letter or underscore, if not, then this is a call
                if string[ind-1] != '_' and not string[ind-1].isalpha():
                    # find the next )
                    ind_end.append(string.find(')', ind))
                else:
                    #remove from ind_begin
                    ind_begin[j] = -1
            # remove -1's
            ind_begin = [ind_begin[i] for i in range(len(ind_begin)) if ind_begin[i] >= 0]
            # remove all hits from string and add them to a variable list
            for j in range(len(ind_begin)):
                begin = ind_begin[j]
                ender = ind_end[j]
                curr_string = string[begin:ender+1]
                string = string[:begin] + ' '*len(curr_string) + string[ender+1:]
                if i == 0:
                    curr_string = curr_string.replace('changed', 'value')
                var_list.append(curr_string)
        return string, var_list

    def rng_conditions_stack(self, for_variables, for_variable_types):
        # find all if statements in body
        # check if rng variable exists in body
        seed = 100
        indentation_depths = self.indentation_depths
        begin_indentation = self.begin_indentation 
        end_indentation = self.end_indentation
        indentation_types = self.indentation_types
        if_statements_indexes = []
        body = self.body.copy()
        len_body = len(body)
        # which if statements are rng'ed if statements
        for i in range(len(indentation_types)):
            if indentation_types[i] == 'if' or indentation_types[i] == 'elif' or indentation_types[i] == 'else':
                # check if there is a rng statement in the if statement
                begin = begin_indentation[i]
                ender = end_indentation[i]
                if self.contains_rng([begin, ender]):
                    if_statements_indexes.append(i)
        
        # analyze line by line which conditions are in the current indentation level
        if len(if_statements_indexes) > 0:
            begin = begin_indentation[if_statements_indexes[0]]
            conditions_by_line = [[] for i in range(len_body)]
            kronbinations_index = self.kronbinations_index
            for i in if_statements_indexes:
                begin = begin_indentation[i]
                ender = end_indentation[i]
                if begin > kronbinations_index:
                    condition_line = body[begin]
                    conditions_string = condition_line.strip()
                    if conditions_string.startswith('if'):
                        conditions_string = conditions_string[2:]
                    elif conditions_string.startswith('elif'):
                        conditions_string = conditions_string[4:]
                    elif conditions_string.startswith('else'):
                        conditions_string = conditions_string[4:]
                    if conditions_string.endswith(':'):
                        conditions_string = conditions_string[:-1]
                    conditions_string = ' '+conditions_string+' '
                    conditions_string, var_list = self.change_kronbinations_calls(conditions_string)
                    conditions_string = conditions_string.replace('(', ' ').replace(')', ' ')  
                    sep_list = [' and ', ' not ', ' or ', ' in ', '==', '!=', '<=', '>=', '<', '>', ' ']
                    for sep in sep_list:
                        conditions_string = conditions_string.replace(sep, ',')
                    # split at commas, keep non empty strings
                    conditions = [c.strip() for c in conditions_string.split(',') if c.strip() != '']  
                    # conditions without indexing
                    conditions_no_index = []
                    for c in conditions:
                        if '[' in c:
                            conditions_no_index.append(c[:c.find('[')])
                        else:
                            conditions_no_index.append(c)
                    # match with for variables # substitute change with var
                    for k, (c, c_no_ind) in enumerate(zip(conditions, conditions_no_index)):
                        if c_no_ind in for_variables:
                            # get index of variable in for_variables
                            index = for_variables.index(c_no_ind)
                            # get type of variable
                            var_type = for_variable_types[index]
                            # if var_type is change -> substitute with val variable
                            if var_type == 'change':
                                # how many_change before this index?
                                n_changes = for_variable_types[:index+1].count('change')
                                # get the val nth var variable where n is n_changes
                                val_var = None
                                curr_n_changes = n_changes
                                for j, v in enumerate(for_variables):
                                    if for_variable_types[j] == 'value':
                                        curr_n_changes -= 1
                                        if curr_n_changes == 0:
                                            val_var = v
                                            break
                                if val_var is None:
                                    # try exchanging with index variable
                                    ind_var = None
                                    curr_n_changes = n_changes
                                    for j, v in enumerate(for_variables):
                                        if for_variable_types[j] == 'index':
                                            curr_n_changes -= 1
                                            if curr_n_changes == 0:
                                                ind_var = v
                                                break
                                    if ind_var is None:
                                        raise Exception('value or index variables corresponding to change variable ('+c+') not found')
                                    else:
                                        conditions[k] = ind_var + c[c.find('['):]
                                        conditions_no_index[k] = ind_var
                                else:
                                    conditions[k] = val_var + c[c.find('['):]
                                    conditions_no_index[k] = val_var
                    # add current conditions to var_dict
                    # add conditions to conditions_by_line for elements begin to end
                    conditions += var_list
                    for k in range(begin+1, ender):
                        conditions_by_line[k] += conditions
            # check line by line for a rng statement
            rng_in_line = [False for i in range(len_body)]
            for i, line in enumerate(body):
                # find variables in line
                if self.contains_rng(i):
                    rng_in_line[i] = True
            # for every indentation check if there is a rng statement and, that isn't in a sub rng statement
            rng_in_if_statement = []
            for i in if_statements_indexes:
                begin = begin_indentation[i]
                ender = end_indentation[i]
                if any(rng_in_line[begin:ender]):
                    # where is the rng statement?
                    rng_index = [i+begin for i, x in enumerate(rng_in_line[begin:ender]) if x]
                    rng_in_if_statement.append(rng_index)
                else:
                    rng_in_if_statement.append([])
            # which indexes are unique to one if statement and not in a sub if statement
            all_rng_indexes = [i for i, x in enumerate(rng_in_line) if x]
            unique_rng_in_if_statement = [[] for i in range(len(if_statements_indexes))]
            for index in all_rng_indexes:
                n_if = len(if_statements_indexes)
                for rng_in_if in rng_in_if_statement[::-1]: # last occurance is the one we want
                    n_if -= 1
                    if index in rng_in_if:
                        unique_rng_in_if_statement[n_if].append(index)
                        break

            #add rng definition to every if condition statement that contains a rng statement 
            for i in range(len(if_statements_indexes)):
                if len(unique_rng_in_if_statement[i]) > 0:
                    begin = begin_indentation[if_statements_indexes[i]] + 1
                    line = body[begin]
                    conditions = conditions_by_line[begin] 
                    # find the number of spaces on the left
                    n_spaces = len(line) - len(line.lstrip(' '))
                    # construct rng object creation
                    #from hashlib import sha1
                    curr_args = ' '*n_spaces + 'curr_args = [' + ', '.join(conditions) + ']'
                    seeder = ' '*n_spaces + "seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)"
                    rnger = ' '*n_spaces + 'rng = np.random.default_rng(seed)'
                    body[begin] = [curr_args, seeder, rnger, line]
        else:
            rng_in_if_statement = []
            unique_rng_in_if_statement = []
            rng_in_line = [False for i in range(len_body)]
            for i, line in enumerate(body):
                # find variables in line
                if self.contains_rng(i):
                    rng_in_line[i] = True
            all_rng_indexes = [i for i, x in enumerate(rng_in_line) if x]

        # check for a rng statement in the rest of the script (later and before)
        # any all_rng_indexes that are not in unique_rng_in_if_statement
        rng_in_else = []
        all_in_if_statements = []
        for i in unique_rng_in_if_statement:
            all_in_if_statements += i
        # unique
        all_in_if_statements = np.array(all_in_if_statements, dtype=int)
        all_in_if_statements = np.unique(all_in_if_statements)
        for index in all_rng_indexes: # the unique remaining indexes
            if not index in all_in_if_statements:
                rng_in_else.append(index)
        # where is the for loop of kronbinations
        kronbinations_index = self.kronbinations_index
        # any rng_in_else that are before the kronbinations_index
        rng_in_else_before = []
        rng_in_else_after = []
        for index in rng_in_else:
            if index < kronbinations_index:
                rng_in_else_before.append(index)
            elif index > kronbinations_index:
                rng_in_else_after.append(index)

        # rng definition at beginning of script
        if len(rng_in_else_before) > 0:
            # find the number of spaces on the left
            line = body[0]
            n_spaces = len(line) - len(line.lstrip(' '))
            # construct rng object creation
            rnger = ' '*n_spaces + 'rng = np.random.default_rng('+str(seed)+')'
            body[0] = [rnger, line]

        # after the kronbinations_index, chek if in between if statements -> make a map of lines in if statements
        in_if = np.zeros(len_body, dtype=bool)
        for i in if_statements_indexes:
            begin = begin_indentation[i]
            ender = end_indentation[i]
            in_if[begin:ender] = True
        i = kronbinations_index+1
        how_many = 0
        while i < len_body:
            # check in_if
            if in_if[i]:
                i += 1
            else:
                beginning = i 
                # find next in_if:
                while i < len_body:
                    if in_if[i]:
                        break
                    i += 1
                ending = i
                # check which rng_in_else_after are in this line
                rng_in_interval = []
                for index in rng_in_else_after:
                    if index >= beginning and index < ending:
                        rng_in_interval.append(index)
                if len(rng_in_interval) > 0:
                    how_many += 1
                    # find the number of spaces on the left
                    line = body[beginning]
                    n_spaces = len(line) - len(line.lstrip(' '))
                    # construct rng object creation
                    curr_args = ' '*n_spaces + 'curr_args = [' + ', '.join(for_variables) + ']'
                    seeder = ' '*n_spaces + "seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)"
                    rnger = ' '*n_spaces + 'rng = np.random.default_rng(seed)'
                    body[beginning] = [curr_args, seeder, rnger, line]
        # expand list of lists into one list
        new_body = []
        for line in body:
            if isinstance(line, list):
                new_body += line
            else:
                new_body.append(line)
        body = new_body
        #dline()
        #for line in body:
        #    print(line)
        return body

    # a function that constructs a function string from the lines of the definition and the new_body 
    def construct_function_string(self):
        # construct function string
        #self.definition, self.body, self.name, self.input_args,
        import_statements = self.import_statements
        input_args = self.input_args
        # remove rng if found
        if 'rng' in input_args:
            input_args.remove('rng')
        # import numpy as np
        function_string = ''
        if not import_statements is None:
            for raw_statement in import_statements:
                if isinstance(raw_statement, str):
                    if raw_statement.startswith('from') or raw_statement.startswith('import'):
                        statement = raw_statement
                    else:
                        statement = 'from ' + raw_statement + ' import *'
                elif isinstance(raw_statement, list):
                    statement = 'import ' + raw_statement[0] + ' as ' + raw_statement[1]
                else:
                    raise Exception('Unknown import statement: use list for "import x as y" or string for "from x import *" or full import statement "x"')
                function_string += statement + '\n'
            if not 'import numpy as np' in import_statements:
                function_string += 'import numpy as np\n'
        else:
            function_string += 'import numpy as np\n'
        function_string += 'from hashlib import sha1\n\n'
        # other func
        for other_func in self.other_func:
            curr_str = inspect.getsource(other_func)
            function_string += curr_str + '\n\n'

        # construct definition
        function_string += 'def ' + self.name + '(' + ', '.join(input_args) + '):\n'
        #function_string = self.definition + '\n'
        for line in self.new_body:
            function_string += line + '\n'
        return function_string

    def import_functions_from_file(self):
        filename = self.file_name
        function_name = self.name
        spec = importlib.util.spec_from_file_location(filename, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        functions = getattr(module, function_name)
        return functions
# Test
#from kronbinations import *

#def gridspace(k, a, b, rng):
#    d = rng.uniform(0, 1.0)
#    for i, v, c in k.kronprod(index=True, changed=True):
#        if c[1] == c[0]:
#            f = rng.uniform(a, b)
#        A[i] = c[0] + f + d
#        e = rng.uniform(a, b)
#        B[i] = c[1] + f + e
#    return A, B

#FM = Kron_Fun_Modifier(gridspace)
#FM.new_body 

#FM. find_all_variables()
#for i in [3]: #range(len(FM.body)):
#    print(FM.variables_in_line(i))
#    dline()
#FM.kronbination_separation()