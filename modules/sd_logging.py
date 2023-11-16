import os
import re
import json
from logging import *
from typing import overload, Any, Optional, Callable, List, Dict, Tuple, Literal, Union
from typing_extensions import Self
from contextlib import contextmanager


'''
One of the primary purposes of this file was to extend the usefulness of the 
logger with regards to adding extensive debugging reporting to discrete chunks
of the application.  Both to make it easier/quicker to add and use, as well as
allowing developers to add, and LEAVE IN their debugging mechanisms, without
resorting to checking in hundreds of commented lines, or cluttering the code
with `if debug:` statements.

Just create a gated logger (or loggers) at the head of your code, and use them
everywhere you need to.  Delete the relevent environment variable, and the logs
go away.  Easy as that.

Below is a breakdown of some of how it works:


log = sd_logger.envConditon('SD_LOG_DEBUG')
log.warn('The following logs will only show up if the `SD_LOG_DEBUG` environment variable is set')
### WARNING  The following logs will only show up if the `SD_LOG_DEBUG` environment variable is set 

log = log.logLevel('DEBUG')
log('Now the default logging level is `info`:  Calling `log()` is the same as calling `log.info()`')
### DEBUG    Now the default logging level is `info`:  Calling `log()` is the same as calling `log.info()` 

log.debug('But calling a log level directly also still works.')        
### DEBUG    But calling a log level directly also still works.

log = log.prefix(f'{__name__}: ')
log('Now the logs will be pefixed.')                                                                                   
### INFO     modules.xxx: Now the logs will be pefixed.

log.logLevel('ERROR')
log('As an error')                                                                                                     
### ERROR    modules.xxx: As an error

log.logLevel()
log('This will execute at the default level of `INFO`')                                                                 
### INFO     modules.xxx: This will execute at the default level of `INFO`

log.logLevel('sdsdyhjfdf') # this will throw an exception
### ValueError: Logging Level `sdsdyhjfdf` does not resolve to a valid logging level.
'''


'''
Another useful trick is this:

for filename in filenames:
    with log.prefix_context(f'Processing {filename}: ') as log:
        log('Loading')
        log.debug(f'size = {file_size}')
        ...
        log.error('Does not exist')
        ...
'''


NoneType = type(None)
_unquoted_strings= [
    lambda s: not re.compile(r'[ =]').search(s), # is num-like
    lambda s: s in ['True', 'False', 'None'] # is bool-like
]


sd_logger: 'SDLogger'


_level_cache = {}


def _getLevel(level:Union[int,str]) -> int | None:
    if level not in _level_cache:
        result = getLevelName(level)
        if type(result) is not int:
            result = getLevelName(result)
        _level_cache[level] = result if type(result) is int else None
    return _level_cache[level]


class _PrefixableLogger:
    
    def prefix(self, prefix:str) -> 'SDLoggerAdapter': 
        return SDLoggerAdapter(self, prefix=prefix)
    
    @contextmanager
    def prefix_context(self, prefix):
        yield self.prefix(prefix)


class _PrefixableLoggerAdapter(_PrefixableLogger):
        
    def __init__(self, logger:Logger, extra = None, *, prefix:str=None):
        self._prefix = prefix
        LoggerAdapter.__init__(self, logger, extra)

    def process(self, msg:str, kwargs:dict[str, Any]) -> str:
        return LoggerAdapter.process(self, f'{self._prefix}{msg}', kwargs)


class _CallableLogger:

    _default_level = None

    def logLevel(self, level:Optional[Union[str, int]]=None) -> Self:
        level = level or 'INFO'
        default_level = _getLevel(level)
        if default_level is None:
            raise ValueError(f'Logging Level `{level}` does not resolve to a valid logging level.')
        self._default_level = default_level
        return self

    @overload
    def __call__(self, msg:str, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, level:int, msg:str=None, *args: Any, **kwargs: Any) -> Any:
        """
        Log 'msg % args' with the integer severity 'level' on `self`.

        Since `sd_logger` will be assigned as `log` at the end of this file,
        this maintains the behavior Python `logging` package, where it has a
        function `log` that calls against the `root` logger.  We (below) define
        all the same functions, just targeting the `sd` logger, and making the
        object callable, we ensure that there is feature parity as best we can.
        """
        if not isinstance(level, int):
            if not self._default_level:
                self.logLevel('INFO')
            if (msg is not None or len(args)):
                args:list = list(args)
                args.insert(0, msg)
            msg = level
            level = self._default_level

        self.log(level, msg, *args, **kwargs)


class _EnvConditionLogger:

    def envConditon(self, *env_flags:list[str]) -> Self | 'DissabledLoger': 
        return self if any([os.environ.get(f'{flag}', None) is not None for flag in env_flags]) else DissabledLoger(self)


class SDLoggerAdapter(_PrefixableLoggerAdapter,LoggerAdapter,_EnvConditionLogger,_CallableLogger): ...


class SDLogger(Logger,_PrefixableLogger,_EnvConditionLogger,_CallableLogger): ...


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


def _print_dict(print_dict:Dict[str,Any], unquoted_string:Dict[str,Callable], print_object:Optional[object]=None) -> None:
    return (print_object.__class__.__name__+'({})' if print_object else '{}').format(' '.join(map(lambda p: _print_pair(p, unquoted_string), print_dict.items())))


setLoggerClass(SDLogger)


sd_logger:SDLogger = getLogger("sd")


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


log:SDLogger = sd_logger