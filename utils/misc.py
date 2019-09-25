import imp
import os

def add_args(args1, args2):
    for k, v in args2.__dict__.items():
        args1.__dict__[k] = v

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    module = imp.load_source(module_name, filename)
    return module, module_name
