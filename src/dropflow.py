# coding: utf-8
"""
Implements incremental rainflow cycle counting algorythm for fatigue analysis
according to section 5.4.4 in ASTM E1049-85 (2011).
"""
from __future__ import division
from collections import deque, defaultdict

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
    
    Attributes
    ----------
    _reversals (list): list of tuples containing the indexes and values of the reversal points.
    _stopper (tuple): tuple containing the index and value of the last point that in the original
        algorithm is threated as a reversal point. Here, since we are working with incremental data,
        we don't know if the last point is a reversal point until the next point is read. Thus, we
        store the last point in the _stopper attribute and update it when the next point is read.
    _closed_cycles (list): list of tuples containing the indexes and values of the closed cycles. This
        attribute is used to store the closed cycles and half cycles that are already formed when the 
        user calls the method extract_all_cycles. Otherwise, the method extract_new_cycles yields only
        the new closed cycles or half cycles.
    _mean (float): mean value of the series.
    _history_length (int): number of points read.
    _idx_last (int): index of the last point read.
    _x_last (float): value of the last point read.
    _x (float): value of the current point.
    _d_last (float): difference between the last two points read.
    """
    def __init__(self) -> None:
        self._reversals = []
        self._stopper = ()
        self._closed_cycles = []

        self._mean = 0
        self._history_length = 0
        self._idx_last = None
        self._x_last = None
        self._x = None
        self._d_last = None
    
    @property
    def reversals(self):
        if self._history_length < 2:
            return []
        return self._reversals + [self._stopper] if self._stopper else self._reversals
        
    def reset(self):
        self._reversals = []
        self._stopper = ()
        self._closed_cycles = []

        self._mean = 0
        self._history_length = 0
        self._idx_last = None
        self._x_last = None
        self._x = None
        self._d_last = None
        
    def add_point(self, x: float, idx: int) -> None:
        """
        Add a point to the series.
        
        Parameters
        ----------
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
            self._idx_last = 0
            
        elif self._history_length == 1:
            self._x = x
            self._d_last = (x - self._x_last)
            self._reversals.append((self._idx_last, self._x_last))
            self._idx_last = idx
        
        else:
            if x == self._x:
                self._idx_last = idx
                return
            
            # Here we decide if the last point is a reversal or not
            d_next = (x - self._x)
            
            if self._d_last * d_next < 0:
                self._reversals.append((self._idx_last, self._x))
            self._x_last, self._x = self._x, x
            self._d_last = d_next
            self._idx_last = idx
            
            # A new point is always a reversal until the following point is read
            self._stopper = (idx, x)
            
    def extract_all_cycles(self, ignore_stopper=False):
        """
        Iterate closed cycles and half cycles.
        In this method we append the closed cycles within the attribute _closed_cycles and pop 
        the relative points from the reversals.
        Instead, the half cycles are yielded and relative points are not popped from the reversals.

        Parameters
        ----------
        ignore_stopper (bool): if True, the last point (stopper) is not considered as a reversal point.
        
        Yields
        ------
        cycle : tuple
            Each tuple contains (range, mean, count, start index, end index).
            Count equals to 1.0 for full cycles and 0.5 for half cycles.
        """ 
        self._reversals.extend([self._stopper]) if self._stopper and not ignore_stopper else None
        
        if len(self._closed_cycles) == 0 and len(self._reversals) < 1:
            print("Not enough samples")
            return []
        
        # Yield already closed cycles
        for cycle in self._closed_cycles:
            yield cycle
                    
        i = 0
        while i < (len(self._reversals) - 2):
            # Form ranges X and Y from the three most recent points
            x1, x2, x3 = self._reversals[i][1], self._reversals[i+1][1], self._reversals[i+2][1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            
            if X < Y:
                # Read the next point
                i += 1
            else:
                if i == 0:
                    # Y contains the starting point
                    # Count Y as one-half cycle and discard the first point
                    self._closed_cycles.append(format_output(self._reversals[i], self._reversals[i+1], 0.5))
                    self._reversals.pop(i)
                    yield self._closed_cycles[-1]
                else:
                    # Count Y as one cycle and discard the peak and the valley of Y
                    self._closed_cycles.append(format_output(self._reversals[i], self._reversals[i+1], 1.0))
                    self._reversals.pop(i)
                    self._reversals.pop(i)
                    yield self._closed_cycles[-1]
                
        else:
            # Count the remaining ranges as one-half cycles 
            for i in range(len(self._reversals) - 1):
                yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                i -= 1
            
            if not ignore_stopper and self._reversals[-1] == self._stopper:
                self._reversals.pop()
            
            
    def extract_new_cycles(self, ignore_stopper=False):
        """
        Iterate closed cycles and half cycles.
        In this method we don't save the closed cycles and we delegate the user to save them.
        Indeed, we just yield the new closed cycles or half cycles.
        
        Parameters
        ----------
        ignore_stopper (bool): if True, the last point (stopper) is not considered as a reversal point.

        Yields
        ------
        cycle : tuple
            Each tuple contains (range, mean, count, start index, end index).
            Count equals to 1.0 for full cycles and 0.5 for half cycles.
        """         
        self._reversals.extend([self._stopper]) if self._stopper and not ignore_stopper else None
        
        if len(self._reversals) < 1:
            print("Not enough samples")
            return []
        
        i = 0
        while i < (len(self._reversals) - 2):
            # Form ranges X and Y from the three most recent points
            x1, x2, x3 = self._reversals[i][1], self._reversals[i+1][1], self._reversals[i+2][1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            
            if X < Y:
                # Read the next point
                i += 1
            else:
                if i == 0:
                    # Y contains the starting point
                    # Count Y as one-half cycle and discard the first point
                    yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                    self._reversals.pop(i)
                else:
                    # Count Y as one cycle and discard the peak and the valley of Y
                    yield format_output(self._reversals[i], self._reversals[i+1], 1.0)
                    self._reversals.pop(i)
                    self._reversals.pop(i)
                
        else:
            # Count the remaining ranges as one-half cycles 
            for i in range(len(self._reversals) - 1):
                yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                i -= 1
                
            if not ignore_stopper and self._reversals[-1] == self._stopper:
                self._reversals.pop()
                

def test():
    series = [
        -1.5, 1.0, -3.0, 10.0, -1.0, 3.0, -8.0, 4.0, -2.0, 6.0,
        -1.0, -4.0, -8.0, 2.0, 1.0, -5.0, 0.0, 2.5, -4.0, 1.0,
        0.0, 2.0, -0.5,
    ]
    expected = [
        (2.5, -0.25, 0.5, 0, 1),
        (4.0, -1.00, 0.5, 1, 2),
        (4.0, 1.00, 1.0, 4, 5),
        (13.0, 3.50, 0.5, 2, 3),
        (6.0, 1.00, 1.0, 7, 8),
        (14.0, -1.00, 1.0, 6, 9),
        (7.0, -1.50, 1.0, 13, 15),
        (1.0, 0.50, 1.0, 19, 20),
        (18.0, 1.00, 0.5, 3, 12),
        (10.5, -2.75, 0.5, 12, 17),
        (6.5, -0.75, 0.5, 17, 18),
        (6.0, -1.00, 0.5, 18, 21),
        (2.5, 0.75, 0.5, 21, 22),
    ]
    
    dropflow = Dropflow()
    
    dropflow.reset()
    for idx, point in enumerate(series):
        dropflow.add_point(x=point, idx=idx)
        
    print(list(dropflow.reversals))
        
    for cycle in dropflow.extract_all_cycles():
        print(cycle in expected, cycle)
        

def test2():
    series = [1, 2]
    dropflow = Dropflow()
    
    dropflow.reset()
    for idx, point in enumerate(series):
        dropflow.add_point(x=point, idx=idx)
        
    print(list(dropflow.reversals))
        
    for cycle in dropflow.extract_all_cycles():
        print(cycle)
    
test()
