import time
import sys


class Timer:
    def __init__(self):
        self.start = time.time()
        self.records = {}
        self.total = 0

    def elapsed(self, reset=True):
        end = time.time()
        res = end - self.start
        if reset:
            self.start = end
        return res

    def add(self, name, t):
        if name not in self.records:
            self.records[name] = t
        else:
            self.records[name] += t

    def record(self, category=None, extra_time=0, reset=True):
        e = self.elapsed(reset)
        if category is None:
            category = sys._getframe(1).f_code.co_name # pylint: disable=protected-access
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += e + extra_time
        self.total += e + extra_time

    def summary(self, min_time=0.05, total=True):
        res = f"{self.total:.2f} " if total else ''
        additions = [x for x in self.records.items() if x[1] >= min_time]
        if not additions:
            return res
        res += " ".join([f"{category}={time_taken:.2f}" for category, time_taken in additions])
        return res

    def dct(self, min_time=0.05):
        return {k: round(v, 2) for k, v in self.records.items() if v >= min_time}

    def reset(self):
        self.__init__()

startup = Timer()
process = Timer()
