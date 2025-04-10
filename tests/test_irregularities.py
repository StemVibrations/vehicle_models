import numpy as np
import matplotlib.pyplot as plt

from UVEC.uvec_ten_dof_vehicle_2D import irregularities

SHOW_PLOTS = False


def test_irregularities():
    """
    Tests the irregularities of the track.
    """

    dist = np.linspace(0, 10, 100)
    irr = np.zeros_like(dist)

    for i, x in enumerate(dist):
        irr[i] = irregularities.calculate_rail_irregularity(x, seed=14)

    with open('./tests/test_data/irregularity.txt', 'r') as f:
        lines = f.read().splitlines()
    lines = np.array([list(map(float, line.split())) for line in lines])

    # Compare the two arrays
    np.testing.assert_almost_equal(irr, lines[:, 1], decimal=5)

    if SHOW_PLOTS:
        plt.plot(dist, irr)
        plt.xlabel('Distance [m]')
        plt.ylabel('Irregularity [m]')
        plt.grid()
        plt.show()


def test_joint():
    """
    Tests the joint irregularities of the track.
    """

    dist = np.linspace(0, 10, 100)
    irr = np.zeros_like(dist)

    # Calculate the irregularities with standard values
    for i, x in enumerate(dist):
        irr[i] = irregularities.calculate_joint_irregularities(x, 5)

    with open('./tests/test_data/hinge_1.txt', 'r') as f:
        lines = f.read().splitlines()
    lines = np.array([list(map(float, line.split())) for line in lines])

    # Compare the two arrays
    np.testing.assert_almost_equal(irr, lines[:, 1], decimal=5)

    # Calculate the irregularities with changed values
    for i, x in enumerate(dist):
        irr[i] = irregularities.calculate_joint_irregularities(x, 5, width_joint=0.5)

    with open('./tests/test_data/hinge_2.txt', 'r') as f:
        lines = f.read().splitlines()
    lines = np.array([list(map(float, line.split())) for line in lines])

    # Compare the two arrays
    np.testing.assert_almost_equal(irr, lines[:, 1], decimal=5)

    if SHOW_PLOTS:
        plt.plot(dist, irr, marker="x")
        plt.xlabel('Distance [m]')
        plt.ylabel('Irregularity [m]')
        plt.grid()
        plt.tight_layout()
        plt.show()
