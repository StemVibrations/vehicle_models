import numpy as np


def calculate_rail_irregularity(x: float,
                                f_min: float = 2,
                                f_max: float = 500,
                                N: int = 2000,
                                Av: float = 0.00002095,
                                omega_c: float = 0.8242,
                                seed: int = 14) -> float:
    """
     Creates rail unevenness following :cite: `Zhang_2001`.

    A summary of default values can be found in :cite: `Podworna_2015`.

    Args:
        - x (float) : position of the wheel [m]
        - f_min (float): minimum spatial frequency for the PSD of the unevenness [1/m] (default 2 1/m)
        - f_max: (float) maximum spatial frequency for the PSD of the unevenness [1/m] (default 500 1/m)
        - N (int): number of frequency increments [-] (default 2000)
        - Av (float): vertical track irregularity parameters [m2 rad/m]  (default 0.00002095 m2 rad/m)
        - omega_c (float): critical wave number [rad/m] (default 0.8242 rad/m)
        - seed: (int) seed for random generator [-] (default 14)

    Returns:
        - irregularity (float): irregularity at the node [m]
    """

    # random generator
    random_generator = np.random.default_rng(seed)

    # define omega range
    omega_max = 2 * np.pi * f_max
    omega_min = 2 * np.pi * f_min
    delta_omega = (omega_max - omega_min) / N

    # for each frequency increment
    omega_n = omega_min + delta_omega * np.arange(N)
    phi = random_generator.uniform(0, 2 * np.pi, N)
    irregularity = np.sum(np.sqrt(4 * spectral(omega_n, Av, omega_c) * delta_omega) * np.cos(omega_n * x - phi))

    return irregularity


def spectral(omega: np.ndarray, Av: float, omega_c: float) -> np.ndarray:
    """
    Computes spectral unevenness

    Args:
        - omega (np.ndarray): wave number [rad/m]
        - Av (float): vertical track irregularity parameters [m2 rad/m]
        - omega_c (float): critical wave number [rad/m]

    Returns:
        - spectral_unevenness (np.ndarray): spectral unevenness [m3 / rad]
    """

    spectral_unevenness = 2 * np.pi * Av * omega_c**2 / ((omega**2 + omega_c**2) * omega**2)
    return spectral_unevenness


def calculate_joint_irregularities(x: float,
                                   location_joint: float,
                                   depth_joint: float = 0.003,
                                   width_joint: float = 1) -> float:
    """
    Creates joint irregularities following :cite: `Kabo_2006`.

    The default values for the depth joint and length joint are taken from :cite: `Kabo_2006`.

    Args:
        - x (float): position of the wheel [m]
        - location_joint (float): absolute location of the joint [m]
        - depth_joint (float): depth of the joint [m] (default 0.003 m)
        - width_joint (float): length of the joint [m] (default 1 m)
    Returns:
        - joint_profile (float): displacement of the joint at the wheel location [m]
    """

    joint_profile = 0

    # find if wheel is on the joint
    if (x >= location_joint - width_joint / 2) & (x <= location_joint):
        joint_profile = -depth_joint * (1 - np.cos(np.pi * (x - (location_joint - width_joint / 2)) / width_joint))
    elif (x > location_joint) & (x <= location_joint + width_joint / 2):
        joint_profile = -depth_joint * (1 + np.cos(np.pi * (x - (location_joint - width_joint / 2)) / width_joint))

    return joint_profile
