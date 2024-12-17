import json
import numpy as np
from typing import Tuple, List

from UVEC.uvec_ten_dof_vehicle_2D.base_model import TrainModel
from UVEC.uvec_ten_dof_vehicle_2D.hertzian_contact import HertzianContact
from UVEC.uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit
from UVEC.uvec_ten_dof_vehicle_2D.irregularities import calculate_rail_irregularity


def uvec(json_string: str) -> str:
    """
    Args:
        - json_string (str): json string containing the uvec data

    Returns:
        - str: json string containing the load data
    """

    # Get the uvec data
    uvec_data = json.loads(json_string)

    # Check if analysis is static
    static = uvec_data["parameters"]["static_initialisation"]

    if static:
        uvec_data = compute_static_solution(uvec_data)
    else:
        uvec_data = compute_dynamic_solution(uvec_data)

    return json.dumps(uvec_data)


def compute_static_solution(uvec_data: dict) -> dict:
    """
    Compute the static solution for the UVEC

    Args:
        - uvec_data (dict): dictionary containing the UVEC data

    Returns:
        - dict: dictionary containing the UVEC data
    """

    u = uvec_data["u"]
    # theta = uvec_data["theta"]
    time_index = uvec_data["time_index"]
    time_step = uvec_data["dt"]
    state = uvec_data["state"]

    # Get the uvec parameters
    parameters = uvec_data["parameters"]

    # initialise the train system
    (_, _, K, F_train), train = initialise(time_index, parameters, state)

    # calculate norm of u vector, gravity is downwards
    gravity_axis = parameters["gravity_axis"]

    u_vertical = [u[uw][gravity_axis] for uw in u.keys()]

    if time_index <= 0:
        state["previous_time"] = 0
        state["previous_time_index"] = time_index

        if "wheel_configuration" in parameters.keys():
            state["current_position"] = [position for position in parameters["wheel_configuration"]]

    if time_index > state["previous_time_index"]:
        state["previous_time"] += time_step

        if "wheel_configuration" in parameters.keys():
            for i in range(len(parameters["wheel_configuration"])):
                state["current_position"][i] = state["current_position"][i] + parameters["velocity"] * time_step

    # check if vertical track irregularity parameter is present and add irregularities if required
    if "irr_parameters" in parameters.keys():
        irregularity_parameters = parameters["irr_parameters"]

        for i in range(len(u_vertical)):
            u_vertical[i] = (calculate_rail_irregularity(state["current_position"][i], **irregularity_parameters) +
                             u_vertical[i])

    # calculate static displacement
    u_static = train.calculate_initial_displacement(K, F_train, u_vertical)

    state["u"] = u_static.tolist()
    state["v"] = np.zeros_like(u_static).tolist()
    state["a"] = np.zeros_like(u_static).tolist()

    # # calculate contact forces
    F_contact = -train.calculate_static_contact_force()

    # calculate unit vector
    aux = {}
    for i, val in enumerate(F_contact):
        aux[i + 1] = [0., (-val).tolist(), 0.]
    uvec_data["loads"] = aux

    state["previous_time_index"] = time_index

    return uvec_data


def compute_dynamic_solution(uvec_data: dict) -> dict:
    """
    Compute the dynamic solution for the UVEC

    Args:
        - uvec_data (dict): dictionary containing the UVEC data

    Returns:
        - dict: dictionary containing the UVEC data
    """

    u = uvec_data["u"]
    # theta = uvec_data["theta"]
    time_index = uvec_data["time_index"]
    time_step = uvec_data["dt"]
    state = uvec_data["state"]

    # Get the uvec parameters
    parameters = uvec_data["parameters"]

    # initialise the train system
    (M, C, K, F_train), train = initialise(time_index, parameters, state)

    # calculate norm of u vector, gravity is downwards
    gravity_axis = parameters["gravity_axis"]

    u_vertical = [u[uw][gravity_axis] for uw in u.keys()]

    if time_index <= 0:
        # calculate static displacement
        u_static = train.calculate_initial_displacement(K, F_train, u_vertical)
        state["u"] = u_static
        state["v"] = np.zeros_like(u_static)
        state["a"] = np.zeros_like(u_static)
        state["previous_time"] = 0
        state["previous_time_index"] = time_index

        if "wheel_configuration" in parameters.keys():
            state["current_position"] = [position for position in parameters["wheel_configuration"]]

    state["u"] = np.array(state["u"])
    state["v"] = np.array(state["v"])
    state["a"] = np.array(state["a"])

    if time_index > state["previous_time_index"]:
        state["previous_time"] += time_step

        if "wheel_configuration" in parameters.keys():
            for i in range(len(parameters["wheel_configuration"])):
                state["current_position"][i] = state["current_position"][i] + parameters["velocity"] * time_step

    # check if vertical track irregularity parameter is present and add irregularities if required
    if "irr_parameters" in parameters.keys():
        irregularity_parameters = parameters["irr_parameters"]

        for i in range(len(u_vertical)):
            u_vertical[i] = (calculate_rail_irregularity(state["current_position"][i], **irregularity_parameters) +
                             u_vertical[i])

    # calculate contact forces
    F_contact = calculate_contact_forces(u_vertical, train.calculate_static_contact_force(), state, parameters, train,
                                         time_index)

    # calculate force vector
    F = F_train
    F[train.contact_dofs] = F[train.contact_dofs] + F_contact

    # scale the force vector based on the amount of initialisation steps
    if "initialisation_steps" in parameters:
        if time_index + 1 < parameters["initialisation_steps"]:
            F = F * (time_index + 1) / parameters["initialisation_steps"]

    # calculate new state
    u_train, v_train, a_train = calculate(state, (M, C, K, F), time_step, time_index)

    state["u"] = u_train.tolist()
    state["v"] = v_train.tolist()
    state["a"] = a_train.tolist()

    # calculate unit vector
    aux = {}
    for i, val in enumerate(F_contact):
        aux[i + 1] = [0., (-val).tolist(), 0.]
    uvec_data["loads"] = aux

    state["previous_time_index"] = time_index

    return uvec_data


def initialise(time_index: int, parameters: dict,
               state: dict) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TrainModel]:
    """
    Initialise the train system

    Args:
        - time_index (int): time index
        - parameters (dict): dictionary containing the parameters
        - state (dict): dictionary containing the state

    Returns:
        - Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TrainModel]: tuple containing the global \
            matrices (M, C, K, F) and the train model
    """

    train = TrainModel()

    train.n_carts = parameters["n_carts"]
    train.cart_inertia = parameters["cart_inertia"]
    train.cart_mass = parameters["cart_mass"]
    train.cart_stiffness = parameters["cart_stiffness"]
    train.cart_damping = parameters["cart_damping"]

    train.bogie_distances = parameters["bogie_distances"]

    train.bogie_inertia = parameters["bogie_inertia"]
    train.bogie_mass = parameters["bogie_mass"]
    train.wheel_distances = parameters["wheel_distances"]

    train.wheel_mass = parameters["wheel_mass"]
    train.wheel_stiffness = parameters["wheel_stiffness"]
    train.wheel_damping = parameters["wheel_damping"]

    train.initialise()

    # set global matrices
    K, C, M, F = train.generate_global_matrices()

    return (M, C, K, F), train


def calculate_contact_forces(u: List, F_static: np.ndarray, state: dict, parameters: dict, train: TrainModel,
                             time_index: int) -> np.ndarray:
    """
    Calculate the contact forces

    Args:
        - u (List): vertical displacement of the wheels
        - F_static (np.ndarray): static contact force
        - state (dict): dictionary containing the state
        - parameters (dict): dictionary containing the parameters
        - train (TrainModel): train model
        - time_index (int): time index

    Returns:
        - np.ndarray: array containing the contact forces
    """

    contact_method = HertzianContact()
    contact_method.contact_coeff = parameters["contact_coefficient"]
    contact_method.contact_power = parameters["contact_power"]

    u_wheel = np.array(state["u"])[train.contact_dofs]

    static_contact_u = contact_method.calculate_contact_deformation(F_static)

    du = u_wheel + static_contact_u - u

    return contact_method.calculate_contact_force(du)


def calculate(state: dict, matrices: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], time_step,
              t) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the new state dynamic

    Args:
        - state (dict): dictionary containing the state
        - matrices (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): tuple containing the global matrices \
            [M, C, K, F]
        - time_step (float): time step
        - t (int): time index

    Returns:
        - tuple: tuple containing the new state
    """

    (M, C, K, F) = matrices
    (u, v, a) = state["u"], state["v"], state["a"]

    solver = NewmarkExplicit()
    return solver.calculate(M, C, K, F, time_step, t, u, v, a)
