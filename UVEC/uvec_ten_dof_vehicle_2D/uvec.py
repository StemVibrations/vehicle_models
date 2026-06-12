import json
import numpy as np
from typing import Tuple, List

from UVEC.uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit
from UVEC.uvec_ten_dof_vehicle_2D.base_model import TrainModel
from UVEC.uvec_ten_dof_vehicle_2D.hertzian_contact import HertzianContact
from UVEC.uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit
from UVEC.uvec_ten_dof_vehicle_2D.irregularities import calculate_rail_irregularity, calculate_joint_irregularities


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

    # if hinge is defined
    if "joint_parameters" in parameters.keys():
        joint_parameters = parameters["joint_parameters"]
        for i in range(len(u_vertical)):
            u_vertical[i] = (calculate_joint_irregularities(state["current_position"][i], **joint_parameters) +
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

        with open("debug.txt", "w") as f:
            pass

        # calculate static displacement
        u_static = train.calculate_initial_displacement(K, F_train, u_vertical)
        state["u"] = u_static
        state["v"] = np.zeros_like(u_static)
        state["a"] = np.zeros_like(u_static)
        state["previous_force"] = train.calculate_static_contact_force().tolist()
        state["residual_factor"] = np.zeros_like(u_vertical).tolist()
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

    # if hinge is defined
    if "joint_parameters" in parameters.keys():
        joint_parameters = parameters["joint_parameters"]
        for i in range(len(u_vertical)):
            u_vertical[i] = (calculate_joint_irregularities(state["current_position"][i], **joint_parameters) +
                             u_vertical[i])

    # F = computeee((M, C, K), state, u_vertical, F_train, train.contact_dofs, time_step, time_index)
    # F, u_next, v_next, a_next = calculate_contact_forces((M, C, K), state, u_vertical, F_train, train.contact_dofs, time_step, time_index)



    n_dof = K.shape[0]
    contact_dof = train.contact_dofs
    # free and prescribed DOFs
    all_indices = np.arange(n_dof)
    free_indices = np.delete(all_indices, contact_dof)

    # partition the stiffness matrix
    K_c = K[:, contact_dof]
    F = K_c.dot(u_vertical)#[contact_dof]
    # F = F - F_train

    # # calculate contact forces
    # F_contact, u_current, v_current, a_current = calculate_contact_forces((M, C, K), F_train, train.calculate_static_contact_force(), state, np.array(u_vertical), train.contact_dofs, time_step, time_index)

    # state["u"] = u_next.tolist()
    # state["v"] = v_next.tolist()
    # state["a"] = a_next.tolist()

    # state["u"] = u_current.tolist()
    # state["v"] = v_current.tolist()
    # state["a"] = a_current.tolist()

    # F = F_train
    # F[train.contact_dofs] = F[train.contact_dofs] + F_contact

    # F_contact = calculate_contact_forces((M, C, K), state, np.array(u_vertical), train.contact_dofs, time_step, time_index)
    # F = F_train
    # F[train.contact_dofs] = F[train.contact_dofs] + F_contact

    # # scale the force vector based on the amount of initialisation steps
    # if "initialisation_steps" in parameters:
    #     if time_index + 1 < parameters["initialisation_steps"]:
    #         F = F * (time_index + 1) / parameters["initialisation_steps"]

    # calculate new state
    u_train, v_train, a_train = calculate(state, (M, C, K, F), time_step, time_index)

    state["u"] = u_train.tolist()
    state["v"] = v_train.tolist()
    state["a"] = a_train.tolist()

    # # update the force vector
    # F_new = (M.dot(a_train) + C.dot(v_train) + K.dot(u_train))
    # F = F[train.contact_dofs]
    # F = train.calculate_static_contact_force() + F_contact
    # F =  -train.calculate_static_contact_force()+F_contact
    # omega = 0.25
    # residual_factor = np.array(state["previous_force"]) - F

    # omega = - (residual_factor.T * (residual_factor - state["residual_factor"])) / np.linalg.norm(residual_factor - state["residual_factor"])**2
    # omega = np.clip(omega, 0, 1)
    # F = (1 - omega) * np.array(state["previous_force"]) + omega * F
    # # F =  np.array(state["previous_force"]) + omega * residual_factor

    # # print(f"Time index: {time_index}, omega: {omega}, F_contact: {F[0]/1e3}")


    state["previous_force"] = F.tolist()
    # state["residual_factor"] = residual_factor.tolist()


    F = F[train.contact_dofs]

    # calculate unit vector
    aux = {}
    for i, val in enumerate(F):
        aux[i + 1] = [0., (val).tolist(), 0.]
    uvec_data["loads"] = aux

    with open("debug.txt", "a+") as f:
        aaa = [str(uvec_data["loads"][i][1]) for i in uvec_data["loads"].keys()]
        f.write(f"{time_index}; {';'.join(aaa)}\n")



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


def calculate_contact_forces(system_matrix: Tuple[np.ndarray, np.ndarray, np.ndarray],
                             state: dict, u_contact: List[float],
                             F_train: np.ndarray, contact_dofs: List[int], time_step: float, time_index: int):
    """
    Calculate the contact forces for the UVEC system

    Args:
        - system_matrix (Tuple[np.ndarray, np.ndarray, np.ndarray]): tuple containing the global matrices [M, C, K]
        - state (dict): dictionary containing the state
        - u_contact (List[float]): list containing the contact displacements
        - F_train (np.ndarray): array containing the external forces
        - contact_dofs (List[int]): list containing the contact degrees of freedom
        - time_step (float): time step
        - time_index (int): time index
    """

    M, C, K = system_matrix

    n_dof = K.shape[0]

    # Get the previous state
    u_prev = np.array(state["u"])
    v_prev = np.array(state["v"])
    a_prev = np.array(state["a"])

    # free and prescribed DOFs
    all_indices = np.arange(n_dof)
    free_indices = np.delete(all_indices, contact_dofs)

    # Partition matrices and vectors from the PREVIOUS time step
    u_f_prev = u_prev[free_indices]
    v_f_prev = v_prev[free_indices]
    a_f_prev = a_prev[free_indices]
    F_f_ext = F_train[free_indices]

    M_ff = M[np.ix_(free_indices, free_indices)]
    M_fp = M[np.ix_(free_indices, contact_dofs)]
    C_ff = C[np.ix_(free_indices, free_indices)]
    C_fp = C[np.ix_(free_indices, contact_dofs)]
    K_ff = K[np.ix_(free_indices, free_indices)]
    K_fp = K[np.ix_(free_indices, contact_dofs)]

    # force vector at prescribed DOFs
    F_eff = F_f_ext - K_fp @ u_contact

    # Solve for the unknown free displacements
    u_f_next = np.linalg.solve(K_ff, F_eff)


    # 4. Assemble the full displacement vector for the current step
    u_next = np.zeros(n_dof)
    u_next[free_indices] = u_f_next
    u_next[contact_dofs] = u_contact

    solver = NewmarkExplicit()
    v_next, a_next = solver.newmark_estimation_v_a(u_next, u_prev, v_prev, a_prev, time_step)

    F_total = M @ a_next + C @ v_next + K @ u_next
    reaction_forces = F_total[contact_dofs] - F_train[contact_dofs]

    return reaction_forces, u_next, v_next, a_next


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
