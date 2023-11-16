import os
import re
import json
from logging import *
from typing import overload, Any, Optional, Callable, List, Dict, Tuple, Literal, Union
from typing_extensions import Self


NoneType = type(None)
_unquoted_strings= [
    lambda s: not re.compile(r'[ =]').search(s), # is num-like
    lambda s: s in ['True', 'False', 'None'] # is bool-like
]


sd_logger: 'SDLogger'


class _PrefixableLogger():
    
    def prefix(self, prefix:str) -> 'SDLoggerAdapter': 
        return SDLoggerAdapter(self, prefix=prefix)


class _PrefixableLoggerAdapter(_PrefixableLogger):
        
    def __init__(self, logger:Logger, extra = None, *, prefix:str):
        self._prefix = prefix
        LoggerAdapter.__init__(self, logger, extra)

    def process(self, msg:str, kwargs:dict[str, Any]) -> str:
        return LoggerAdapter.process(self, f'{self._prefix}{msg}', kwargs)


class _EnvConditionLogger():

    def envConditon(self, *env_flags:list[str]) -> Self | 'DissabledLoger': 
        return self if any([os.environ.get(f'{flag}', None) is not None for flag in env_flags]) else DissabledLoger(self)


class SDLoggerAdapter(LoggerAdapter,_PrefixableLoggerAdapter,_EnvConditionLogger): ...


class SDLogger(Logger,_PrefixableLogger,_EnvConditionLogger):

    def __call__(self, level:Union[int,str], msg:str, *args: Any, **kwargs: Any) -> Any:
        """
        Log 'msg % args' with the integer severity 'level' on `self`.
        """
        self.log(level, msg, *args, **kwargs)


class DissabledLoger(SDLoggerAdapter):

    def isEnabledFor(self, _level) -> Literal[False]:
        return False


def _print_val(value:Any, unquoted_string:Optional[List[Callable]]=None) -> str:
    assert isinstance(unquoted_string, (list, NoneType)), f'optional parameter `unquoted_string` must be of type `list`, got type `{type(unquoted_string)}`'
    if type(value) is str:
        if not any(check(value) for check in _unquoted_strings) or (unquoted_string and any(check(value) for check in unquoted_string)):
            return json.dumps(value)
    return f'{value}'


def _print_pair(pair:Tuple[str,Any], unquoted_string:Optional[List[Callable]]=None) -> str:
    return f"{_print_val(pair[0])}={_print_val(pair[1], unquoted_string=unquoted_string)}"


@overload
def print_dict(_print_dict:Dict[str,Any]) -> None: ...
def print_dict(_print_object:object=None, _dict:Dict[str,Any]=None, _unquoted_string:Optional[List[Callable]]=None, /, **kwargs) -> None:
    if _unquoted_string:
        if callable(_unquoted_string):
            _unquoted_string = [_unquoted_string]
    if not _dict:
        if kwargs:
            _dict = kwargs
        elif type(_print_object) is dict:
            _dict = _print_object
            _print_object = None
    assert type(_dict) is dict, f'required parameter `print_dict` must be of type `dict`, got type `{type(print_dict)}`'
    assert isinstance(_print_object, (object, NoneType)), f'optional parameter `print_object` must be and instance of `object` or `None`'
    return _print_dict(_dict, print_object=_print_object, unquoted_string=_unquoted_string)

print(type(None))

def _print_dict(print_dict:Dict[str,Any], unquoted_string:Dict[str,Callable], print_object:Optional[object]=None) -> None:
    return (print_object.__class__.__name__+'({})' if print_object else '{}').format(' '.join(map(lambda p: _print_pair(p, unquoted_string), print_dict.items())))


setLoggerClass(SDLogger)


sd_logger = getLogger("sd")


def critical(msg, *args, **kwargs):
    """
    Log a message with severity 'CRITICAL' on the SD logger.
    """
    sd_logger.critical(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    """
    Don't use this function, use critical() instead.
    """
    critical(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """
    Log a message with severity 'ERROR' on the SD logger.
    """
    sd_logger.error(msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
    """
    Log a message with severity 'ERROR' on the SD logger, with exception
    information.
    """
    error(msg, *args, exc_info=exc_info, **kwargs)


def warning(msg, *args, **kwargs):
    """
    Log a message with severity 'WARNING' on the SD logger.
    """
    sd_logger.warning(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    """
    warnings.warn("The 'warn' function is deprecated, "
        "use 'warning' instead", DeprecationWarning, 2)
    """
    warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the SD logger.
    """
    sd_logger.info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """
    Log a message with severity 'DEBUG' on the SD logger.
    """
    sd_logger.debug(msg, *args, **kwargs)


@overload
def log(level, msg, *args, **kwargs):
    """
    Log 'msg % args' with the integer severity 'level' on the SD logger.
    """
    sd_logger.log(level, msg, *args, **kwargs)


log = sd_logger