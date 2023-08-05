#!/usr/bin/env python3
"""Integrate polynomial vector"""


def poly_integral(poly, C=0):
    """Integrate poly with constant factor C"""
    if not all(type(c) in (float, int) for c in poly) or type(C) != int:
        return None
    integral = [c/i if c % i != 0 else c//i for i, c in enumerate(poly, 1)]
    while len(integral) > 0 and integral[-1] == 0:
        integral.pop()
    return [C] + integral
