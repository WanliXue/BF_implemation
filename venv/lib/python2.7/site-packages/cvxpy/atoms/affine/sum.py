"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


def sum(expr, axis=None, keepdims=False):
    """Wrapper for Sum class.
    """
    if isinstance(expr, list):
        return __builtins__['sum'](expr)
    else:
        return Sum(expr, axis, keepdims)


class Sum(AxisAtom, AffAtom):
    """ Summing the entries of an expression.

    Attributes
    ----------
    expr : CVXPY Expression
        The expression to sum the entries of.
    """

    def __init__(self, expr, axis=None, keepdims=False):
        super(Sum, self).__init__(expr, axis=axis, keepdims=keepdims)

    def numeric(self, values):
        """Sums the entries of value.
        """
        return np.sum(values[0], axis=self.axis, keepdims=self.keepdims)

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Sum the linear expression's entries.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        obj = lu.sum_entries(arg_objs[0], shape=shape,
                             axis=data[0], keepdims=data[1])
        return (obj, [])
