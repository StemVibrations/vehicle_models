import numpy as np
from spring_damper_model.newmark_solver import NewmarkExplicit

import pytest
import unittest
class TestSolver(unittest.TestCase):


    def test_newmark_explicit(self):
        """
        Checks if Newmark solver gives the same results as the example from bathe


        :return:
        """

        M = np.array([[2, 0], [0, 1]])
        K = np.array([[6, -2], [-2, 4]])
        C = np.array([[0, 0], [0, 0]])
        F = np.zeros(2)
        F[1] = 10

        n_steps = 11
        t_step = 0.28
        t_total = n_steps * t_step

        time = np.linspace(
            0, t_total, int(np.ceil((t_total - 0) / t_step) + 1)
        )

        u = np.zeros(2)
        v = np.zeros(2)
        a = np.zeros(2)

        solver = NewmarkExplicit()

        all_u = np.zeros((len(time), len(u)))
        all_v = np.zeros((len(time), len(v)))
        all_a = np.zeros((len(time), len(a)))

        for i in range(len(time)):

            u,v,a = solver.calculate(M, C, K, F, t_step, i, u, v, a)

            all_u[i,:] = u
            all_v[i,:] = v
            all_a[i,:] = a

        np.testing.assert_array_almost_equal(
            np.round(all_u, 2),
            np.round(
                np.array(
                    [
                        [0.00673, 0.364],
                        [0.0505, 1.35],
                        [0.189, 2.68],
                        [0.485, 4.00],
                        [0.961, 4.95],
                        [1.58, 5.34],
                        [2.23, 5.13],
                        [2.76, 4.48],
                        [3.00, 3.64],
                        [2.85, 2.90],
                        [2.28, 2.44],
                        [1.40, 2.31],
                    ]
                ),
                2,
            ),
        )
