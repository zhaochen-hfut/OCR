# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:09:02 2020

@author: zhaochen

"""

import time
import inspect
from functools import wraps
from typing import Tuple

def arg_value(arg_name, func, args, kwargs):
    
    if arg_name in kwargs:
        return kwargs[arg_name]
    
    i = func.__code__.co_varnames.index(arg_name)
    
    if i < len(args):
        return args[i]
    
    return inspect.signature(func).parameters[arg_name].default

def logger(begin_message: str = None, log_args: Tuple[str] = None, end_message: str = None,
           log_time: bool = True):
    
    def logger_decorator(func):
        @wraps(func)
        def decorate(*args, **kwargs):
            if begin_message is not None:
                print(begin_message, end = '\n')
            if log_args is not None:
                arg_logs = [ arg_name + '=' + str( arg_value(arg_name, func, args, kwargs)) for arg_name in log_args]
                print('parameters: ' + ', '.join(arg_logs))
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            spent_time = end_time - start_time
            
            if  end_message is not None:
                print(end_message)
            
            if log_time:
                print('耗时' + str(spent_time) + '秒')
            
            return result
        return decorate
    return logger_decorator       