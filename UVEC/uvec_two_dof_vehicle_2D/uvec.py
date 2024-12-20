import json
import numpy as np


def uvec(json_string: str) -> str:
    """
    Args:
        - json_string (str): json string containing the uvec data

    Returns:
        - str: json string containing the load data
    """

    gravity_axis = 1

    # Get the uvec data
    uvec_data = json.loads(json_string)

    # load the data
    u = uvec_data["u"]
    # theta = uvec_data["theta"]
    time_index = uvec_data["time_index"]
    time_step = uvec_data["dt"]
    state = uvec_data["state"]

    # Get the uvec parameters
    # mass_1 = uvec_data["parameters"]["m1"]  # mass of the top
    mass_2 = uvec_data["parameters"]["m2"]  # mass of the bottom
    stiffness = uvec_data["parameters"]["k"]
    damping = uvec_data["parameters"]["c"]

    u_beam = [u[uw][gravity_axis] for uw in u.keys()]

    # if first two time step initialise the variables. need to be two because we need the acceleration and velocity
    if time_index <= 1:
        # two dof system
        state["u"] = 0
        state["v"] = 0
        state["a"] = 0
        state["u_g_previous"] = 0
        state["v_g_previous"] = 0
        state["a_g_previous"] = 0
        state["u_beam"] = 0

    # compute the dof for the system given the u_beam
    state["u_beam"] = u_beam[0]
    state = compute_dofs(uvec_data["parameters"], state, time_step, time_index)

    force = mass_2 * state["a_g_previous"] + damping * state["v_g_previous"] + stiffness * (state["u_beam"] -
                                                                                            state["u"]) - mass_2 * 9.81

    # Set the load data
    uvec_data['loads'] = {1: [0, -force, 0]}
    uvec_data["state"] = state

    return json.dumps(uvec_data)


def compute_dofs(parameters: dict, state: dict, delta_t: float, time_index: int) -> dict:
    """
    Compute the solution of a two dof system using the Newmark method

    Args:
        - parameters (dict): dictionary containing the parameters of the system
        - state (dict): dictionary containing the state of the system
        - delta_t (float): time step
        - time_index (int): time index

    Returns:
        - dict: dictionary containing the state of the system
    """

    m1 = parameters["m1"]
    # m2 = parameters["m2"]
    k1 = parameters["k"]
    c1 = parameters["c"]
    external_displacement = state["u_beam"]
    u = np.array(state["u"])
    v = np.array(state["v"])
    a = np.array(state["a"])

    if time_index < 1:
        return state

    beta = 0.25
    gamma = 0.5

    M = m1
    K = k1
    C = c1

    # external force
    f_ext = k1 * external_displacement + M * 9.81

    a1 = 1 / (beta * delta_t**2)
    a2 = 1 / (beta * delta_t)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / (beta * delta_t)
    a5 = (gamma / beta) - 1
    a6 = (gamma / beta - 2) * delta_t / 2

    # newmark method
    K_till = K + C * a4 + M * a1

    m_part = u * a1 + v * a2 + a * a3
    c_part = u * a4 + v * a5 + a * a6
    m_part = M * m_part
    c_part = C * c_part

    # external force
    force_ext = f_ext + m_part + c_part

    # solve
    uu = force_ext / K_till

    # velocity calculated through Newmark relation
    vv = (uu - u) * a4 - v * a5 - a * a6
    # acceleration calculated through Newmark relation
    aa = (uu - u) * a1 - v * a2 - a * a3

    state["u"] = uu
    state["v"] = vv
    state["a"] = aa
    state["u_beam"] = external_displacement

    new_d = external_displacement
    new_v = (external_displacement - state["u_g_previous"]) / delta_t
    new_a = (new_v - state["v_g_previous"]) / delta_t
    state["u_g_previous"] = new_d
    state["v_g_previous"] = new_v
    state["a_g_previous"] = new_a

    return state
