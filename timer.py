#!/usr/bin/env python

import time


def timer(func) -> float:
    def timer_inner(*args, **kwargs):
        t0: float = time.time()
        func(*args, **kwargs)
        t1: float = time.time()
        print(f"Elapsed time {t1 - t0} in function {func.__name__}")

    return timer_inner
