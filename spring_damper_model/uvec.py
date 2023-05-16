import numpy as np
from typing import Union, List, Dict

from spring_damper_model.base_model import TrainModel
def uvec(u: np.ndarray, theta: np.ndarray, time_step: float, state: Union[Dict, List] = None,
         parameters: Union[Dict, List] = None):
    """
    Calculate the unit vector in the direction from point u to point v.

    Parameters
    ----------
    u: displacement vector at wheels
    theta: rotation vector at wheels
    time_step: time step for the integration
    state: optional state of the train system
    parameters: optional parameters
    """


    initialise(parameters)

    # force in x-y-z
    force = [0, 20, 0]

    # constant load / wheel
    loads = [force, force, force, force]

    return loads

def initialise(parameters):
    """
    Initialise the train system
    """


    train = TrainModel()

    train.n_carts = parameters["n_carts"]
    train.n_bogies = parameters["n_bogies"]
    train.n_wheels = parameters["n_wheels"]

    train.cart_intertia = parameters["cart_intertia"]
    train.cart_mass = parameters["cart_mass"]
    train.cart_stiffness = parameters["cart_stiffness"]
    train.cart_damping = parameters["cart_damping"]

    train.bogie_distances = parameters["bogie_distances"]

    train.bogie_intertia = parameters["bogie_intertia"]
    train.bogie_mass = parameters["bogie_mass"]
    train.wheel_distances = parameters["wheel_distances"]



    # number of wheels
    n_wheels = 4

    # number of degrees of freedom
    n_dof = 2 * n_wheels

    # mass matrix

    M = train.generate_mass_matrix()
    D = train.generate_damping_matrix()
    K = train.generate_stiffness_matrix()

    F = train.generate_force_vector()

    # initial displacement
    u_0 = np.zeros(n_dof)

    # initial velocity
    v_0 = np.zeros(n_dof)

    # initial acceleration
    a_0 = np.zeros(n_dof)

    # initial state
    state = [u_0, v_0, a_0]

    # parameters
    parameters = [M, K]

    return state, parameters

