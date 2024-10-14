# coding: utf-8
"""
Implements incremental rainflow cycle counting algorythm for fatigue analysis
according to section 5.4.4 in ASTM E1049-85 (2011).
"""
from __future__ import division
from collections import deque, defaultdict


def _get_round_function(ndigits=None):
    if ndigits is None:
        def func(x):
            return x
    else:
        def func(x):
            return round(x, ndigits)
    return func


def format_output(point1, point2, count):
            i1, x1 = point1
            i2, x2 = point2
            rng = abs(x1 - x2)
            mean = 0.5 * (x1 + x2)
            return rng, mean, count, i1, i2


class Dropflow:
    """
    Class of the incremental rainflow cycle counting algorithm.
    The name Dropflow represents the idea of dropping a point at time instead of the 
    entire "rain" of points at once.
    """
    def __init__(self) -> None:
        self._reversals = deque()
        self._closed_cycles = []

        self._mean = 0
        self._history_length = 0
        self._x_last = None
        self._x = None
        self._d_last = None
    
    @property
    def reversals(self):
        return self._reversals
        
    def reset(self):
        self._reversals = deque()
        self._closed_cycles = []

        self._mean = 0
        self._history_length = 0
        self._x_last = None
        self._x = None
        self._d_last = None
        
    def add_point(self, x: float, idx: int) -> None:
        """
        Add a point to the series.
        
        Args:
            x (float): value of the point
            idx (int): index of the point
        """
        self._check_reversal(x, idx)
        self._mean = (self._mean * (self._history_length) + x) / (self._history_length + 1)
        self._history_length += 1
    
    def _check_reversal(self, x: float, idx: int) -> None:
        """
        Check if the provided point is a reversal point.

        A reversal point is a point in the series at which the first derivative
        changes sign. Reversal is undefined at the first (last) point because the
        derivative before (after) this point is undefined. The first and the last
        points are treated as reversals.

        Parameters
        ----------
        x (float): value of the point
        idx (int): index of the point
        """
        if self._history_length == 0:
            self._x_last = x
            
        elif self._history_length == 1:
            self._x = x
            self._d_last = (x - self._x_last)
        
        else:
            # Here we decide if the last point is a reversal or not
            d_next = (x - self._x)
            if not self._d_last * d_next < 0:
                self._reversals.pop()
                self._d_last = (x - self._x_last)
            else:     
                self._x_last = self._x
                self._d_last = d_next
            self._x = x
        
        # A new point is always a reversal until the following point is read
        self._reversals.append((idx, x))
        
    def extract_all_cycles(self):
        """
        Iterate closed cycles and half cycles.
        In this method we append the closed cycles within the attribute _closed_cycles and pop 
        the relative points from the reversals.
        Instead, the half cycles are yielded and relative points are not popped from the reversals.

        Yields
        ------
        cycle : tuple
            Each tuple contains (range, mean, count, start index, end index).
            Count equals to 1.0 for full cycles and 0.5 for half cycles.
        """ 
        if len(self._closed_cycles) == 0 and len(self._reversals) < 2:
            print("Not enough samples")
            return []
        
        # Yield already closed cycles
        for cycle in self._closed_cycles:
            yield cycle
        
        for i in range(len(self._reversals) - 2):
            # Form ranges X and Y from the three most recent points
            x1, x2, x3 = self._reversals[i][1], self._reversals[i+1][1], self._reversals[i+2][1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            
            if X < Y:
                # Read the next point
                continue
            else:
                # Count Y as one cycle and discard the peak and the valley of Y
                self._closed_cycles.append(format_output(self._reversals[i], self._reversals[i+1], 1.0))
                self._reversals.pop(i)
                self._reversals.pop(i+1)
                i -= 1
                yield self._closed_cycles[-1]
                
        else:
            # Count the remaining ranges as one-half cycles
            for i in range(len(self._reversals) - 1):
                yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                i -= 1
                
    def extract_new_cycles(self):
        """
        Iterate closed cycles and half cycles.
        In this method we don't save the closed cycles and we delegate the user to save them.
        Indeed, we just yield the new closed cycles or half cycles.

        Yields
        ------
        cycle : tuple
            Each tuple contains (range, mean, count, start index, end index).
            Count equals to 1.0 for full cycles and 0.5 for half cycles.
        """ 
        if len(self._reversals) < 2:
            print("Too few samples")
            return []
        
        for i in range(len(self._reversals) - 2):
            # Form ranges X and Y from the three most recent points
            x1, x2, x3 = self._reversals[i][1], self._reversals[i+1][1], self._reversals[i+2][1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            
            if X < Y:
                # Read the next point
                continue
            else:
                # Count Y as one cycle and discard the peak and the valley of Y
                yield format_output(self._reversals[i], self._reversals[i+1], 1.0)
                self._reversals.pop(i)
                self._reversals.pop(i+1)
                i -= 1
                
        else:
            # Count the remaining ranges as one-half cycles
            for i in range(len(self._reversals) - 1):
                yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                i -= 1
                