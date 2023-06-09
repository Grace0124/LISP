"""
6.1010 Spring '23 Lab 12: LISP Interpreter
"""
#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)

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


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
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


def get_char_type(char):
    """Return type of input."""
    operators = {"+", "*", "/", "(", ")"}
    if char.isdigit():
        return "num"
    elif char.isalpha():
        return "let"
    elif char == "-":
        return "min"
    elif char == ".":
        return "dot"
    elif char in operators:
        return "opr"
    elif char == " ":
        return "spa"
    elif char == ";":
        return "com"
    else:
        return "let"


table_map = {"num": 0, "let": 1, "min": 2, "opr": 3, "dot": 4, "spa": 5, "com": 6}
state_table = {
    "Start": ["Num_1", "Letter", "Minus", "Operand", "Num_2", "Space", "Comment"],
    "Num_1": ["Num_1", "EmitT", "Minus", "EmitT", "Num_2", "EmitT", "EmitT"],
    "Num_2": ["Num_2", "EmitT", "EmitT", "EmitT", "Num_2", "EmitT", "EmitT"],
    "Minus": ["Num_1", "EmitT", "EmitT", "EmitT", "Num_2", "EmitT", "EmitT"],
    "Letter": ["Letter", "Letter", "Letter", "EmitT", "EmitT", "EmitT", "EmitT"],
    "Operand": ["EmitT", "EmitT", "EmitT", "EmitT", "EmitT", "EmitT", "EmitT"],
    "Space": ["Start", "Start", "Start", "Start", "Start", "Start", "Start"],
    "Comment": ["Start", "Start", "Start", "Start", "Start", "Start", "Start"],
    "EmitT": ["Start", "Start", "Start", "Start", "Start", "Start", "Start"],
}


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    lines = source.split("\n")
    state = "Start"
    result = []
    curr = ""

    for line in lines:
        next_line = False
        stiter = iter(line)
        c = next(stiter, None)
        while c:
            if next_line is True:
                continue
            ct = get_char_type(c)
            if ct is None:
                return []
            new_state = state_table[state][table_map[ct]]
            if new_state == "Space":
                c = next(stiter, None)
                new_state = "Start"
            elif new_state == "Comment":
                next_line = True
                break
            elif new_state == "EmitT":
                if curr:
                    result.append(curr)
                curr = ""
                new_state = "Start"
            else:
                curr += c
                c = next(stiter, None)

            state = new_state
        if curr:
            result.append(curr)
            curr = ""

    return result


def check_valid(tokens):
    """
    Checks if tokens if a list and has an equal number of
    left and right parentheses.
    """
    if not isinstance(tokens, list):
        return False
    lp_count = 0
    rp_count = 0
    stiter = iter(tokens)
    c = next(stiter, None)
    while c:
        if c == "(":
            lp_count += 1
        elif c == ")":
            rp_count += 1
        elif " " in c:
            return False
        c = next(stiter, None)
    if lp_count != rp_count:
        return False
    return True


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    if not check_valid(tokens):
        raise SchemeSyntaxError

    def parse_expression(index):
        """
        Takes in an index for tokens (list) and returns a
        pair of values:
            - the expression found starting at index
            - the index where the expression ends plus 1
        """
        outside_list = []
        token = tokens[index]
        if token == "(":
            index += 1
            while index < len(tokens) and tokens[index] != ")":
                val, index = parse_expression(index)
                outside_list.append(val)
            if tokens[index] != ")":
                raise SchemeSyntaxError
            return (outside_list, index + 1)
        else:
            if tokens[index] == ")":
                raise SchemeSyntaxError
            return (number_or_symbol(token), index + 1)

    parsed_expression, next_index = parse_expression(0)
    if next_index != len(tokens):
        raise SchemeSyntaxError
    return parsed_expression


def multiply(args):
    prod = args[0]
    for val in args[1:]:
        prod *= val
    return prod


def divide(args):
    numerator = args[0]
    for val in args[1:]:
        numerator /= val
    return numerator


def equal(args):
    val = args[0]
    for arg in args[1:]:
        if val != arg:
            return False
    return True


def less_than(args):
    prev_value = args[0]
    for arg in args[1:]:
        if arg >= prev_value:
            return False
        prev_value = arg
    return True


def leq(args):
    prev_value = args[0]
    for arg in args[1:]:
        if arg > prev_value:
            return False
        prev_value = arg
    return True


def greater_than(args):
    prev_value = args[0]
    for arg in args[1:]:
        if arg <= prev_value:
            return False
        prev_value = arg
    return True


def geq(args):
    prev_value = args[0]
    for arg in args[1:]:
        if arg < prev_value:
            return False
        prev_value = arg
    return True


def and_func(args, frame):
    for arg in args:
        try:
            if evaluate(arg, frame) is False:
                return False
        except:
            return False
    return True


def or_func(args, frame):
    for arg in args:
        try:
            if evaluate(arg, frame) is True:
                return True
        except:
            continue
    return False


def not_func(arg):
    if len(arg) != 1:
        raise SchemeEvaluationError("Received too many arguments")
    if arg[0]:
        return False
    else:
        return True


######################
# Built-in Functions #
######################


def define(args, frame):
    """
    Inputs: args (len 2 list), frame
    args should be a list with two values: [var_name, var_value]
    Returns the value of the variable.
    """
    frame.bind_var(args[0], args[1])
    return args[1]


def create_function(params, body, frame):
    """
    Takes in a list of parameters, a body (expression to be evaluated),
    and a frame.
    Creates a new Function object and returns it."""
    new_func = User_Function(params, body, frame)
    return new_func


def call_cons(args):
    """Creates a new Pair object and returns it."""
    if len(args) != 2:
        raise SchemeEvaluationError("Incorrect Number of Arguments.")
    new_con = Pair(args[0], args[1])
    return new_con


def car(args):
    """Returns the first element in a Pair object."""
    if len(args) != 1:
        raise SchemeEvaluationError("Empty list")
    pair = args[0]
    if not isinstance(pair, Pair):
        raise SchemeEvaluationError("Pair object not passed through.")
    return pair.car


def cdr(args):
    """Returns the second element in a Pair object."""
    if len(args) != 1:
        raise SchemeEvaluationError("Empty list")
    pair = args[0]
    if not isinstance(pair, Pair):
        raise SchemeEvaluationError("Pair object not passed through.")
    return pair.cdr


def list_func(args):
    """
    takes in a sequence of numbers and creates a list.
    (list) -> nil
    (list 1) -> (1, nil)
    (list 1 2) -> (1, (2, nil))
    """
    if len(args) == 0:
        return "nil"
    args.reverse()
    curr = Pair(args[0], "nil")
    for arg in args[1:]:
        curr = Pair(arg, curr)
    return curr


def is_list(args):
    """
    True if args[0] is a list. False otherwise.
    """
    if args[0] == "nil":
        return True
    if isinstance(args[0], Pair):
        lst = args[0]
        while lst.cdr != "nil":
            lst = lst.cdr
            if not isinstance(lst, Pair):
                return False
        return True

    return False


def length_list(args):
    """Returns length of a linked list."""
    if is_list([args[0]]) == False:
        raise SchemeEvaluationError("Not a linked list.")
    curr_list = args[0]
    if curr_list == "nil":
        return 0
    next_list = curr_list.cdr
    length = 1
    while next_list != "nil":
        next_list = next_list.cdr
        length += 1
    return length


def return_index(args):
    """args[0] should be a list. args[1] should be an index."""
    index = args[1]
    lst = args[0]
    curr = 0

    if lst == "nil":
        raise SchemeEvaluationError("List is None.")

    if not isinstance(lst.cdr, Pair):
        if index == 0:
            return lst.car
        else:
            raise SchemeEvaluationError("Pair Index out of range.")

    if index >= length_list([lst]):
        raise SchemeEvaluationError("List Index out of range.")

    if not isinstance(lst, Pair) and index == 0:
        return lst.car

    while curr != args[1]:
        lst = lst.cdr
        curr += 1

    return lst.car


def append_list(args):
    """takes an arbitrary number of lists and returns a new list
    representing the concatenation of these lists"""

    if args == []:
        return "nil"

    if args[0] == "nil":
        rest = args[1:]
        return append_list(rest)
    else:
        if not isinstance(args[0], Pair):
            raise SchemeEvaluationError("Cannot append to non-list.")
        first = args[0].car
        rest = []
        rest.append(args[0].cdr)
        rest.extend(args[1:])
        return Pair(first, append_list(rest))


def map_func(args):
    """Takes in args, a list consisting of a function and a list.
    Applies the function to each element of the list and returns a list"""

    func = args[0]
    lst = args[1]

    if lst == "nil":
        return "nil"

    if not isinstance(lst, Pair):
        raise SchemeEvaluationError

    return Pair(func([lst.car]), map_func([func, lst.cdr]))


def filter_func(args):
    """args = [func, list of arguments].
    Returns a new list consisting of all args where func(args) is True."""
    func = args[0]
    lst = args[1]

    if lst == "nil":
        return "nil"

    if not isinstance(lst, Pair):
        raise SchemeEvaluationError

    if func([lst.car]) == True:
        return Pair(lst.car, filter_func([func, lst.cdr]))
    else:
        return filter_func([func, lst.cdr])


def reduce(args):
    """args = [function, list, initial value].
    Successively applies a function to the elements in the list.
    Example: (reduce * (list 9 8 7) 1)
    1 is multiplied by 9, then 8, then 7. Returns 504.
    """
    func = args[0]
    lst = args[1]
    init = args[2]

    if lst == "nil":
        return init

    curr = func([init, lst.car])
    return reduce([func, lst.cdr, curr])


def begin(args, frame):
    """args: list of arbitrary length.
    Evaluates all arguments. Only returns the last argument."""
    # print(f'begin frame: {frame}')
    for arg in args:
        result = evaluate(arg, frame)
    return result


def evaluate_file(file, frame=None):
    if frame is None:
        frame = Frame(global_frame)
    f = open(file, "r")
    z = f.read()
    return evaluate(parse(tokenize(z)), frame)


def delete(var, frame):
    """Takes in a variable name and a frame.
    Removes the variable from the current frame and returns its value.
    If no variable found, raises a SchemeNameError"""
    if var in frame.bindings:
        return_val = frame.bindings[var]
        del frame.bindings[var]
        return return_val
    else:
        raise SchemeNameError("Var not in current frame.")


def let(args, body, parent_frame):
    frame = Frame(parent_frame)
    for pair in args:
        frame.bindings[pair[0]] = evaluate(pair[1], parent_frame)
    return evaluate(body, frame)


def set_bang(var, exp, frame):
    """
    Takes in a variable name, an expression, and a frame.
    Evaluates the expression in the current frame.
    Finds the nearest enclosing frame in which var is defined and updates its value.
    If no var is found, raises a SchemeNameError

    """

    # print(f'{var=}, {exp=}')
    def find_var_frame(var, frame):
        if frame is None:
            raise SchemeNameError
        if var in frame.bindings:
            return frame
        else:
            return find_var_frame(var, frame.parent)

    var_frame = find_var_frame(var, frame)
    try:
        x = evaluate(exp, frame)
    except:
        x = evaluate(exp, var_frame)

    var_frame.bindings[var] = x
    return x


and_or_funcs = {
    "and": and_func,
    "or": or_func,
}

scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": multiply,
    "/": divide,
    "not": not_func,
    "define": define,
    "lambda": create_function,
    "equal?": equal,
    ">": less_than,
    ">=": leq,
    "<": greater_than,
    "<=": geq,
    "cons": call_cons,
    "cdr": cdr,
    "car": car,
    "list": list_func,
    "list?": is_list,
    "length": length_list,
    "list-ref": return_index,
    "append": append_list,
    "map": map_func,
    "filter": filter_func,
    "reduce": reduce,
}


booleans = {"#t": True, "#f": False}


##########
# Frames #
##########
class Frame:
    num = 0

    def __init__(self, parent):
        self.parent = parent
        self.bindings = {}
        self.num = Frame.num
        Frame.num += 1

    def bind_var(self, var, value):
        self.bindings[var] = value

    def get_parent(self):
        return self.parent

    def get_bindings(self):
        return self.bindings.copy()

    def return_var(self, var):
        if var in self.bindings:
            return self.bindings[var]
        else:
            if self.parent:
                return self.parent.return_var(var)
        raise SchemeNameError

    def __str__(self):
        return f"Frame {self.num}"


class User_Function:
    def __init__(self, params, body, frame):
        self.params = params
        self.body = body
        self.frame = frame

    def __call__(self, args):
        if len(args) != len(self.params):
            raise SchemeEvaluationError("Incorrect number of arguments")
        # make a new frame whose parent is the function's enclosing frame
        new_frame = Frame(self.frame)
        # in the new frame, bind function's parameters to the arguments passed through
        for var, val in zip(self.params, args):
            new_frame.bindings[var] = val
        return evaluate(self.body, new_frame)

    def __str__(self):
        return f"Function. Params = {self.params}. Body = {self.body}"


built_in_frame = Frame(None)
for k, v in scheme_builtins.items():
    built_in_frame.bind_var(k, v)
built_in_frame.bind_var("nil", "nil")

global_frame = Frame(built_in_frame)


class Pair:
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def __str__(self):
        return f"({self.car}, {self.cdr})"


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
    if tree == []:
        raise SchemeEvaluationError("Tree is empty.")
    if frame is None:
        frame = Frame(global_frame)

    if isinstance(tree, list):
        if isinstance(tree[0], list):
            func = evaluate(tree[0], frame)
            # print(func)
            return func([evaluate(x, frame) for x in tree[1:]])

        elif tree[0] == "define":
            func = frame.return_var("define")
            lambda_func = frame.return_var("lambda")
            if isinstance(tree[1], list):
                # ex: ['define', ['fib', 'n'], [body]]
                return func(
                    [tree[1][0], lambda_func(tree[1][1:], tree[2], frame)], frame
                )
            else:
                # ex: ['define', 'x', ['+' 5 6]]
                result = func([tree[1], evaluate(tree[2], frame)], frame)
                for i in range(2, len(tree) - 1):
                    result = func(
                        [evaluate(result, frame), evaluate(tree[i + 1], frame)], frame
                    )
                return result

        elif tree[0] == "lambda":
            lambda_func = frame.return_var("lambda")
            return lambda_func(tree[1], tree[2], frame)

        elif tree[0] == "if":
            if evaluate(tree[1], frame):
                return evaluate(tree[2], frame)
            else:
                return evaluate(tree[3], frame)

        elif tree[0] in and_or_funcs:
            func = and_or_funcs[tree[0]]
            return func(tree[1:], frame)

        elif tree[0] == "begin":
            return begin(tree[1:], frame)

        elif tree[0] == "del":
            return delete(tree[1], frame)

        elif tree[0] == "let":
            return let(tree[1], tree[2], frame)

        elif tree[0] == "set!":
            return set_bang(tree[1], tree[2], frame)

        else:
            if isinstance(tree[0], (float, int)) and len(tree) > 1:
                raise SchemeEvaluationError("List of ints cannot be evaluated.")
            func = frame.return_var(tree[0])
            if func:
                return func([evaluate(x, frame) for x in tree[1:]])
            else:
                print("mid of evaluate")
                raise SchemeNameError("No func.")

    elif tree in booleans:
        return booleans[tree]
    elif isinstance(tree, (int, float)):
        return tree
    else:
        result = frame.return_var(tree)
        if result is not None:
            return frame.return_var(tree)
        print(tree)
        print("end of evaluate")
        raise SchemeNameError("No variable found.")


def result_and_frame(tree, frame=None):
    if frame is None:
        frame = Frame(global_frame)
    return (evaluate(tree, frame), frame)


def range_func(start, stop, step):
    if start > stop:
        return "nil"
    return Pair(start, range_func(start + step, stop, step))


def repl(verbose=False):
    """
    Read in a single line of user input, evaluate the expression, and print
    out the result. Repeat until user inputs "QUIT"

    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """
    import traceback

    _, frame = result_and_frame(["+"])  # make a global frame

    while True:
        input_str = input("in> ")
        if input_str == "QUIT":
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if verbose:
                print("expression>", expression)
            output = evaluate(expression, frame)
            print("  out>", output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print("Error>", repr(e))


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    # for file in sys.argv:
    #     evaluate_file(file, frame)
    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    repl(False) #use repl(True) for more detailed output
    pass
