########
# REPL #
########

import os
import sys
import traceback
from cmd import Cmd

try:
    import readline
except:
    readline = None


def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.  Not guaranteed to work in all cases, but maybe in most?
    """
    plat = sys.platform
    supported_platform = plat != "Pocket PC" and (
        plat != "win32" or "ANSICON" in os.environ
    )
    # IDLE does not support colors
    if "idlelib" in sys.modules:
        return False
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


class SchemeREPL(Cmd):
    """
    Class that implements a Read-Evaluate-Print Loop for our Scheme
    interpreter.
    """

    history_file = os.path.join(os.path.expanduser("~"), ".6101_scheme_history")

    if supports_color():
        prompt = "\001\033[96m\002in>\001\033[0m\002 "
        value_msg = "  out> \033[92m\033[1m%s\033[0m"
        error_msg = "  \033[91mEXCEPTION!! %s\033[0m"
    else:
        prompt = "in> "
        value_msg = "  out> %s"
        error_msg = "  EXCEPTION!! %s"

    # fmt: off
    keywords = {
        "define", "lambda", "if", "equal?", "<", "<=", ">", ">=", "and", "or",
        "del", "let", "set!", "+", "-", "*", "/", "#t", "#f", "not", "cons",
        "list", "cat", "cdr", "list-ref", "length", "append", "begin",
    }
    # fmt: on

    def __init__(self, lab_module, use_frames=False, verbose=False, global_frame=None):
        self.verbose = verbose
        self.use_frames = use_frames
        self.module = lab_module
        if use_frames:
            self.global_frame = (
                global_frame
                if global_frame is not None
                else self.module.make_initial_frame()
            )
        Cmd.__init__(self)

    def preloop(self):
        try:
            if readline and os.path.isfile(self.history_file):
                readline.read_history_file(self.history_file)
        except:
            pass

    def postloop(self):
        if readline:
            readline.set_history_length(10_000)
            readline.write_history_file(self.history_file)

    def completedefault(self, text, line, begidx, endidx):
        try:
            bound_vars = set(self.global_frame)
        except:
            bound_vars = set()
        return sorted(i for i in (self.keywords | bound_vars) if i.startswith(text))

    def onecmd(self, line):
        if line in {"EOF", "quit", "QUIT"}:
            print()
            print("bye bye!")
            return True

        elif not line.strip():
            return False

        try:
            token_list = self.module.tokenize(line)
            if self.verbose:
                print("tokens>", token_list)
            expression = self.module.parse(token_list)
            if self.verbose:
                print("expression>", expression)
            args = [expression]
            if self.use_frames:
                args.append(self.global_frame)
            output = self.module.evaluate(*args)
            print(self.value_msg % output)
        except self.module.SchemeError as e:
            if self.verbose:
                traceback.print_tb(e.__traceback__)
                print(self.error_msg.replace("%s", "%r") % e)
            else:
                print(self.error_msg % e)

        return False

    completenames = completedefault

    def cmdloop(self, intro=None):
        while True:
            try:
                Cmd.cmdloop(self, intro=None)
                break
            except KeyboardInterrupt:
                print("^C")
