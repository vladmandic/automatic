import itertools
import os
from collections import UserDict
from threading import Thread
from queue import Queue, Empty
from os import PathLike
from typing import Callable, Dict, Iterator, List, Optional, Union, Iterable


FilePathList = List[str]
FilePathIterator = Iterator[str]
DirectoryPathList = List[str]
DirectoryPathIterator = Iterator[str]
DirectoryList = List['Directory']
DirectoryIterator = Iterator['Directory']
DirectoryCollection = Dict[str, 'Directory']
ExtensionFilter = Callable
ExtensionList = list[str]
RecursiveType = Union[bool,Callable]


def real_path(directory_path:str) -> Union[str, None]:
    try:
        return os.path.abspath(os.path.expanduser(directory_path))
    except Exception:
        pass
    return None


def threaded_walker(directory, queue):
    try:
        scandir_it = os.scandir(directory)
    except OSError:
        ...
    if scandir_it:
        with scandir_it:
            while True:
                path = False
                try:
                    path = next(scandir_it, False)
                except OSError:
                    ...
                if path is False:
                    break
                queue.put(path)
    queue.put(True)


class DirectoryWalker():
    def __init__(self, directory):
        self.directory = directory
        self.__complete = False
        self.paths: list[PathLike] = []
        self.queue = Queue()
        self.thread = Thread(target=threaded_walker, daemon=True, args=[self.directory, self.queue])
        self.mtime = self.directory.live_mtime
        self.thread.start()


    def is_stale(self, mtime_compare:float):
        return mtime_compare and mtime_compare != self.mtime


    def is_complete(self, blocking = True) -> bool:
        path = None
        while not self.__complete:
            try:
                path = self.queue.get_nowait()
            except Empty:
                if path is None and blocking:
                    path = self.queue.get()
                else:
                    break
            if path is True:
                self.__complete = True
            elif path:
                self.paths.append(path)
            elif blocking:
                raise GeneratorExit('Unknown Error, empty path, somehow')

        return self.__complete


    @property
    def files(self) -> Iterable[str]:
        i = 0
        while not self.is_complete() or i < len(self.paths):
            i += 1
            try:
                if self.paths[i-1].is_dir():
                    continue
            except OSError:
                ...
            yield self.paths[i-1].path


    @property
    def directories(self) -> Iterable[str]:
        i = 0
        while not self.is_complete() or i < len(self.paths):
            try:
                if self.paths[i].is_dir():
                    yield self.paths[i].path
            except OSError:
                ...
            i += 1


class Directory(PathLike):

    def __init__(self, filepath):
        self.path = filepath
        self.mtime = None
        self.__walker = None
        self.__files = set()
        self.__directories = set()
        self.__sync()


    def __fspath__(self) -> str:
        return f'{self.path}'


    def __sync(self):
        if self.is_directory:
            if (not self.__walker and self.is_stale) or (self.__walker and self.__walker.is_stale(self.mtime)):
                self.__walker = DirectoryWalker(self)


    def __is_complete(self) -> bool:
        self.__sync()
        if self.__walker and self.__walker.is_complete(False):
            walker = self.__walker
            self.__walker = None
            self._update({
                'mtime': walker.mtime,
                'files': set(walker.files),
                'directories': set(walker.directories),
            })
        return self.__walker is None


    @property
    def files(self) -> Union[set[str], Iterable[str]]:
        return self.__files if self.__is_complete() else self.__walker.files


    @property
    def directories(self) -> Union[set[str], Iterable[str]]:
        return self.__directories if self.__is_complete() else self.__walker.directories


    @property
    def is_stale(self) -> bool:
        return not self.mtime or self.mtime != self.live_mtime


    @property
    def is_directory(self) -> bool:
        return is_directory(self.path)


    @property
    def live_mtime(self) -> float:
        return os.path.getmtime(self.path)

    @classmethod
    def from_dict(cls, dict_object: Dict) -> 'Directory':
        directory = cls.__new__(cls)
        directory._update(dict_object)
        return directory

    @property
    def dict(self) -> Dict:
        return {
            'path': self.path,
            'directories': set(self.directories),
            'files': set(self.files),
            'mtime': self.mtime
        }

    def clear(self) -> None:
        self._update({
            'path': None,
            'mtime': float(),
            'files': [],
            'directories': []
        })

    def update(self, source_directory: 'Directory') -> 'Directory':
        if source_directory is not self:
            self._update(source_directory.dict)
        return self

    def _update(self, source:Dict) -> None:
        assert source.get('path', None) is None or source.path == self.path, f'When updating a directory, the paths must match.  Attemped to update Directory `{self.path}` with `{source.path}`'
        for dead_directory in self.__directories:
            if dead_directory not in source.directories:
                delete_cached_directory(dead_directory)
        self.mtime = source.get('mtime')
        self.__files = set(source.get('files'))
        self.__directories = set(source.get('directories'))


class DirectoryCache(UserDict, DirectoryCollection):
    def __delattr__(self, directory_path: str) -> None:
        directory: Directory = get_directory(directory_path, fetch=False)
        if directory:
            map(delete_cached_directory, directory.directories)
            directory.clear()
        del self.data[directory_path]


def get_directory(directory_or_path: str, /, fetch:bool=True) -> Union[Directory, None]:
    if isinstance(directory_or_path, Directory):
        if directory_or_path.is_directory:
            return directory_or_path
        else:
            directory_or_path = directory_or_path.path
    directory_or_path = real_path(directory_or_path)
    if not cache_folders.get(directory_or_path, None):
        if fetch:
            directory = Directory(directory_or_path)
            if directory.is_directory:
                cache_folders[directory_or_path] = directory
            else:
                print(f'Not Dir: {directory_or_path}')
    return cache_folders[directory_or_path] if directory_or_path in cache_folders else None


def cached_walk(directory_path, recurse:RecursiveType=True) -> Iterable[Directory]:
    directory_path = get_directory(directory_path)
    if not directory_path:
        return
    yield directory_path
    if recurse:
        for child_directory in directory_path.directories:
            if os.path.basename(child_directory).startswith('models--'):
                continue
            if callable(recurse) and not recurse(child_directory):
                continue
            yield from cached_walk(child_directory, recurse=recurse)


def delete_cached_directory(directory_path:str) -> bool:
    global cache_folders # pylint: disable=W0602
    if directory_path in cache_folders:
        del cache_folders[directory_path]


def is_directory(dir_path:str) -> bool:
    return dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path)


def directory_mtime(directory_path:str, /, recursive:RecursiveType=True) -> float:
    return float(max(0, *[directory.mtime for directory in get_directories(directory_path, recursive=recursive)]))


def unique_directories(directories:DirectoryPathList, /, recursive:RecursiveType=True) -> DirectoryPathIterator:
    '''Ensure no empty, or duplicates'''
    '''If we are going recursive, then directories that are children of other directories are redundant'''
    ''' @todo this is incredibly inneficient.  the hit is small, but it is ugly, no? '''
    directories = sorted(unique_paths(directories), reverse=True)
    while directories:
        directory = directories.pop()
        yield directory
        if not recursive:
            continue
        _directory = os.path.join(directory, '')
        child_directory = None
        while directories and directories[-1].startswith(_directory):
            if not callable(recursive) or not child_directory:
                directories.pop()
                continue
            child_directory = directories[-1][len(directory):]
            if child_directory:
                next_directory = _directory
                if not callable(recursive):
                    _remove_directory = next_directory
                else:
                    for sub_directory in child_directory.split(os.path.sep):
                        next_directory = os.path.join(next_directory, sub_directory)
                        if recursive(next_directory):
                            _remove_directory = os.path.join(next_directory, '')
                            break
                while _remove_directory and directories:
                    _d = directories.pop()
                    if not directories[-1].startswith(_remove_directory):
                        del _remove_directory


def unique_paths(directory_paths:DirectoryPathList) -> DirectoryPathIterator:
    realpaths = (real_path(directory_path) for directory_path in filter(bool, directory_paths))
    return {real_directory_path: True for real_directory_path in filter(bool, realpaths)}.keys()


def get_directories(*directory_paths: DirectoryPathList, fetch:bool=True, recursive:RecursiveType=True) -> DirectoryCollection:
    directory_paths = unique_directories(directory_paths, recursive=recursive)
    directories = (get_directory(directory_path, fetch=fetch) for directory_path in directory_paths)
    return filter(bool, directories)


def directory_files(*directories_or_paths: Union[DirectoryPathList, DirectoryList], recursive: RecursiveType=True) -> FilePathIterator:
    return itertools.chain.from_iterable(
        itertools.chain(
            directory_object.files,
            []
            if not recursive
            else itertools.chain.from_iterable(
                directory_files(directory, recursive=recursive)
                for directory
                in filter(
                    bool,
                    map(get_directory, filter(((bool if recursive else False) if not callable(recursive) else recursive), directory_object.directories))
                )
            )
        )
        for directory_object
        in filter(bool, map(get_directory, directories_or_paths))
    )


def extension_filter(ext_filter: Optional[ExtensionList]=None, ext_blacklist: Optional[ExtensionList]=None) -> ExtensionFilter:
    if ext_filter:
        ext_filter = [*map(str.upper, ext_filter)]
    if ext_blacklist:
        ext_blacklist = [*map(str.upper, ext_blacklist)]
    def filter_functon(fp:str):
        return (not ext_filter or any(fp.upper().endswith(ew) for ew in ext_filter)) and (not ext_blacklist or not any(fp.upper().endswith(ew) for ew in ext_blacklist))
    return filter_functon


def not_hidden(filepath: str) -> bool:
    return not os.path.basename(filepath).startswith('.')


def filter_files(file_paths: FilePathList, ext_filter: Optional[ExtensionList]=None, ext_blacklist: Optional[ExtensionList]=None) -> FilePathIterator:
    return filter(extension_filter(ext_filter, ext_blacklist), file_paths)


def list_files(*directory_paths:DirectoryPathList, ext_filter: Optional[ExtensionList]=None, ext_blacklist: Optional[ExtensionList]=None, recursive:RecursiveType=True) -> FilePathIterator:
    return filter_files(itertools.chain.from_iterable(
        directory_files(directory, recursive=recursive)
        for directory in get_directories(*directory_paths, recursive=recursive)
    ), ext_filter, ext_blacklist)


cache_folders = DirectoryCache({})
