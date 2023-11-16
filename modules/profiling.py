
import io
import pstats
import cProfile
from contextlib import contextmanager


def dump_profile(profile: cProfile.Profile, msg: str) -> str:
    profile.disable()
    stream = io.StringIO()
    ps = pstats.Stats(profile, stream=stream)
    ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(15)
    profile = None
    lines = stream.getvalue().split('\n')
    lines = [line for line in lines if '<frozen' not in line and '{built-in' not in line and '/logging' not in line and '/rich' not in line]
    return f'Profile {msg}:', '\n'.join(lines)


def print_profile(profile: cProfile.Profile, msg: str) -> None:
    try:
        from rich import print # pylint: disable=redefined-builtin
    except Exception:
        pass
    print(dump_profile(profile, msg))


def new_profiler():
    _cProfile = cProfile.Profile()
    _cProfile.enable()
    return _cProfile


@contextmanager
def profile_context(message:str, profile:bool=True):
    try:
        if profile:
            _cProfile = new_profiler()
        yield
    finally:
        if profile:
            print_profile(_cProfile, message)