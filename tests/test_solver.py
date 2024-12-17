import numpy as np

from UVEC.uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit


def test_Newmark():
    """
    Test the Newmark solver against the example from Bathe
    """

    # example from bathe
    M = np.array([[2, 0], [0, 1]])
    K = np.array([[6, -2], [-2, 4]])
    C = np.array([[0, 0], [0, 0]])
    F = np.zeros((2, 13))
    F[1, :] = 10
    u0 = np.zeros((2))
    v0 = np.zeros((2))
    a0 = np.zeros((2))

    dt = 0.28
    nb_steps = 12
    time = np.linspace(0, 0.28 * 12, 12 + 1)

    solver = NewmarkExplicit()
    res_u = []
    res_u.append(u0)
    for i in range(nb_steps):
        u, v, a = solver.calculate(M, C, K, F[:, i], dt, i, u0, v0, a0)
        res_u.append(u)
        u0 = np.copy(u)
        v0 = np.copy(v)
        a0 = np.copy(a)

    np.testing.assert_array_almost_equal(
        np.round(np.array(res_u), 2),
        np.round(
            np.array([
                [0, 0],
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
            ]),
            2,
        ),
    )
