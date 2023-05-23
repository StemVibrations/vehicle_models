import numpy as np
from typing import Union, List, Dict

from spring_damper_model.base_model import TrainModel
from spring_damper_model.hertzian_contact import HertzianContact
def uvec(u: np.ndarray, theta: np.ndarray, time_step: float, time_index: int, state: Union[Dict, List] = None,
         parameters: Union[Dict, List] = None):
    """
    Calculate the unit vector in the direction from point u to point v.

    Parameters
    ----------
    u: displacement vector at wheels
    theta: rotation vector at wheels
    time_step: time step for the integration
    time_index: time index
    state: optional state of the train system
    parameters: optional parameters
    """

    # initialise the train system
    (M, C, K, F_train), train = initialise(time_index, parameters, state)

    # calculate norm of u vector
    u_norm = np.linalg.norm(u, axis=1)

    # calculate static displacement
    u_static = train.calculate_initial_displacement(K, F_train, u_norm)

    if time_index <= 0:
        state["u"] = u_static
        state["v"] = np.zeros_like(u_static)
        state["a"] = np.zeros_like(u_static)

    # calculate contact forces
    F_contact = calculate_contact_forces(u_norm, train.calculate_static_contact_force(),
                                         state, parameters, train, time_index)

    # calculate force vector
    F = F_train
    F[train.contact_dofs] += F_contact

    # calculate new state
    u_train, v_train, a_train = calculate(parameters,state,(M, C, K, F), time_step, time_index)

    state["u"] = u_train
    state["v"] = v_train
    state["a"] = a_train

    return (-F[train.contact_dofs]).tolist()


def initialise(time_index, parameters, state):
    """
    Initialise the train system
    """


    train = TrainModel()

    train.n_carts = parameters["n_carts"]


    train.cart_intertia = parameters["cart_intertia"]
    train.cart_mass = parameters["cart_mass"]
    train.cart_stiffness = parameters["cart_stiffness"]
    train.cart_damping = parameters["cart_damping"]

    train.bogie_distances = parameters["bogie_distances"]

    train.bogie_intertia = parameters["bogie_intertia"]
    train.bogie_mass = parameters["bogie_mass"]
    train.wheel_distances = parameters["wheel_distances"]

    train.wheel_mass = parameters["wheel_mass"]
    train.wheel_stiffness = parameters["wheel_stiffness"]
    train.wheel_damping = parameters["wheel_damping"]


    train.initialise()

    # mass matrix
    M = train.generate_mass_matrix()
    C = train.generate_damping_matrix()
    K = train.generate_stiffness_matrix()

    F = train.generate_force_vector()

    for i in range(M.shape[0]):
        if np.isclose(M[i,i],0):
            M[i,i] = 1
            K[i,i] = 1
            C[i,i] = 1

            F[i] = 0

    return (M, C, K, F), train


def calculate_contact_forces(u, F_static, state, parameters, train, time_index):


    contact_method = HertzianContact()
    contact_method.contact_coeff = parameters["contact_coefficient"]
    contact_method.contact_power = parameters["contact_power"]

    u_wheel = state["u"][train.contact_dofs]

    static_contact_u = contact_method.calculate_contact_deformation(F_static)

    du = u_wheel + static_contact_u - u

    return contact_method.calculate_contact_force(du)



def calculate(parameters, state, matrices,time_step ,t):
    (M, C, K, F) = matrices

    (u, v, a) = state["u"], state["v"], state["a"]

    from spring_damper_model.newmark_solver import NewmarkExplicit

    solver = NewmarkExplicit()
    return solver.calculate(M,C,K,F,time_step, t, u,v,a)


if __name__ == "__main__":

    parameters = {
        "n_carts": 1,
        "cart_intertia": 1,
        "cart_mass": 1,
        "cart_stiffness": 1e7,

        "cart_damping": 1e5,
        "bogie_distances": [-1,1],
        "bogie_intertia": 1,
        "bogie_mass": 1,
        "wheel_distances": [-1,1],
        "wheel_mass": 1,
        "wheel_stiffness": 1e7,
        "wheel_damping": 1e5,



        "contact_coefficient": 9.1e-7,
        "contact_power": 3/2
    }

    state = {
        "u": [],
        "v": [],
        "a": []
    }

    F_array= []

    n_time_steps = 5

    u_track = np.zeros((4,3))
    u_track[:,1] = np.array([1,1,1,1])

    for t in range(n_time_steps):

        F = uvec(u_track, np.array([0,0,0,0]), 0.0001, t, state, parameters)
        F_array.append(F)

    F_array = np.array(F_array)

    import matplotlib.pyplot as plt
    plt.plot(F_array[:,0])
    plt.show()



    tmp=1+1



    pass