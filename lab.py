"""
6.101 Lab:
LISP Interpreter Part 1
"""

#!/usr/bin/env python3

# import doctest # optional import
# import typing  # optional import
# import pprint  # optional import

import sys

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    result = []
    curr = ""
    in_comment = False
    for char in source:
        if in_comment:
            if char != "\n":
                continue
            else:
                in_comment = False
        elif char == ";":
            in_comment = True
        elif char in ["(", ")", " ", "\n"]:
            if curr:
                result.append(curr)
                curr = ""
            if char in ["(", ")"]:
                result.append(char)
        else:
            curr += char

    if curr:
        result.append(curr)
    return result


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    def parse_expression(curr_li, idx):
        x = number_or_symbol(tokens[idx])
        while idx < len(tokens) and tokens[idx] != ")":
            x = number_or_symbol(tokens[idx])
            if x == "(":
                new_li, idx = parse_expression([], idx + 1)
                curr_li.append(new_li)
            else:
                curr_li.append(x)
                idx += 1
        return curr_li, idx + 1

    parsed_expr, _ = parse_expression([], 0)
    return parsed_expr[0]


######################
# Built-in Functions #
######################


def calc_sub(*args):
    if len(args) == 1:
        return -args[0]
    else:
        first_num, *rest_nums = args
        return first_num - scheme_builtins["+"](*rest_nums)


def calc_mul(*args):
    result = 1
    for num in args:
        result *= num
    return result


def calc_div(*args):
    if len(args) == 1:
        return args[0]
    else:
        first_num, *rest_nums = args
        return first_num / scheme_builtins["*"](*rest_nums)


scheme_builtins = {
    "+": lambda *args: sum(args),
    "-": calc_sub,
    "*": calc_mul,
    "/": calc_div,
    "define": "define",
    "lambda": "lambda",
}


##############
# Frames #
##############
class Frame(object):
    def __init__(self, parent):
        self.parent = parent
        self.variables = {}

    def add_var(self, name, val):
        self.variables[name] = val

    def get_var(self, name):
        if name in self.variables:
            return self.variables[name]
        elif self.parent:
            return self.parent.get_var(name)
        else:
            raise SchemeNameError()

    def contains(self, name):
        if name in self.variables:
            return True
        elif self.parent:
            return self.parent.contains(name)
        else:
            return False

    def __str__(self):
        return f"Variables in frame: {self.variables}"


def make_initial_frame():
    global_frame = Frame(None)
    global_frame.variables = scheme_builtins
    return Frame(global_frame)


##############
# User-defined Functions #
##############


class Function(object):
    def __init__(self, frame, parameters, body):
        self.frame = frame  # Frame
        self.parameters = parameters  # list
        self.body = body

    def __call__(self, *args):
        if len(args) != len(self.parameters):
            raise SchemeEvaluationError(
                f"num of args {len(args)} and num of params {len(self.parameters)} differ"
            )

        func_frame = Frame(self.frame)

        for i, val in enumerate(args):
            func_frame.add_var(self.parameters[i], val)
        return evaluate(self.body, func_frame)

    def __str__(self):
        return f"function: {self.parameters=}. {self.body=}"


##############
# Evaluation #
##############


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """

    if frame is None:
        frame = make_initial_frame()

    if isinstance(tree, (int, float)):  # number
        return tree

    elif isinstance(tree, str):  # variable name
        return frame.get_var(tree)

    elif isinstance(tree, list):

        if tree[0] == "define":
            if isinstance(tree[1], list):  # func short-form
                func_name, func_args = tree[1][0], tree[1][1:]
                func_body = tree[2]
                new_func = Function(frame, func_args, func_body)
                frame.add_var(func_name, new_func)
                return new_func
            else:
                var_name = tree[1]
                var_val = evaluate(tree[2], frame)
                frame.add_var(var_name, var_val)
                return var_val

        elif tree[0] == "lambda":  # create a user-defined function
            f = Function(frame, tree[1], tree[2])
            return f

        else:  # call a function

            func = evaluate(tree[0], frame)

            if not callable(func):
                raise SchemeEvaluationError()

            args = []
            for x in tree[1:]:
                val = evaluate(x, frame)
                args.append(val)
            return func(*args)


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    # import schemerepl
    # schemerepl.SchemeREPL(sys.modules[__name__], use_frames=True, verbose=True).cmdloop()
    # t = tokenize("(
    print(parse(tokenize("(define (square x) (* x x))")))
