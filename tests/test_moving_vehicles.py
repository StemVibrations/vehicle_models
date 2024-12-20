import json
import matplotlib.pyplot as plt
import numpy as np

from UVEC.uvec_ten_dof_vehicle_2D.uvec import uvec
from UVEC.uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit

from tests.utils import UtilsFct

INSPECT_RESULTS = False


class TestSpringDamperModel():

    def test_spring_damper_model(self):
        """
        Tests multiple moving vehicles on a simply supported beam.
        """

        json_input_file = {
            "dt": 0.001,
            "loads": {
                "1": [0, 0, 0]
            },
            "parameters": {
                "n_carts": 2,
                "cart_inertia": (1128.8e3) / 2,
                "cart_mass": (50e3) / 2,
                "cart_stiffness": 2708e3,
                "cart_damping": 64e3,
                "bogie_distances": [-9.95, 9.95],
                "bogie_inertia": (0.31e3) / 2,
                "bogie_mass": (6e3) / 2,
                "wheel_distances": [-1.25, 1.25],
                "wheel_mass": 1.5e3,
                "wheel_stiffness": 4800e3,
                "wheel_damping": 0.25e3,
                "contact_coefficient": 9.1e-5,
                "contact_power": 1.0,
                "gravity_axis": 1,  # 0 = x, 1 = y, 2 = z
                "static_initialisation": False,
            },
            "state": {
                "a": [],
                "u": [],
                "v": []
            },
            "t": 0,
            "theta": {
                "1": [0.0, 0.0, 0.0],
                "2": [0.0, 0, 0.0],
                "3": [0.0, 0, 0.0],
                "4": [0.0, 0, 0.0],
                "5": [0.0, 0, 0.0],
                "6": [0.0, 0, 0.0],
                "7": [0.0, 0, 0.0],
                "8": [0.0, 0, 0.0],
            },
            "time_index": 0,
            "u": {
                "1": [0.0, 0, 0.0],
                "2": [0.0, 0, 0.0],
                "3": [0.0, 0, 0.0],
                "4": [0.0, 0, 0.0],
                "5": [0.0, 0, 0.0],
                "6": [0.0, 0, 0.0],
                "7": [0.0, 0, 0.0],
                "8": [0.0, 0, 0.0],
            }
        }

        # set vehicle location parameters
        loc_vehicle = [0.0, 2.5, 19.9, 22.4, 25.4, 27.9, 47.8, 50.3]
        velocity = 100 / 3.6

        # Euler beam parameters
        n_beams = 500
        length_beam = 250
        E = 2.87e9
        I = 290
        L = length_beam / n_beams
        rho = 2303
        A = 1
        omega_1 = 0
        omega_2 = 0

        # Create the euler beam structure
        euler_beam_structure = UtilsFct.create_simply_supported_euler_beams(n_beams, E, I, L, rho, A, omega_1, omega_2)

        u_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        # u_structure[0::2] = 1
        v_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        a_structure = np.zeros(euler_beam_structure.K_global.shape[0])

        # initialize time integration solver for simply supported beam
        solver = NewmarkExplicit()

        # set time integration parameters
        n_steps = 401
        time = np.linspace(0, 7.1, n_steps)
        dt = time[1] - time[0]

        json_input_file["dt"] = dt

        # initialize arrays to store results
        all_u_beam = []
        all_u_bogie = []

        # loop over time steps
        for t in range(n_steps - 1):

            # get vertical displacement at vehicle location on beam
            u_vert = UtilsFct.get_result_at_x_on_simply_supported_euler_beams(u_structure, euler_beam_structure,
                                                                              loc_vehicle)
            for i in range(len(u_vert)):
                json_input_file["u"][f"{i+1}"][1] = u_vert[i]
            json_input_file["time_index"] = t

            # call uvec model and retrieve force at wheel, this is what is tested.
            return_json = uvec(json.dumps(json_input_file))
            json_input_file = json.loads(return_json)

            F_vehicle = [np.array(json_input_file["loads"][str(i + 1)][1]) for i in range(len(loc_vehicle))]

            # set force at vehicle location on beam
            F_at_structure = np.zeros(euler_beam_structure.K_global.shape[0])
            for i, f in enumerate(F_vehicle):
                F_at_structure = F_at_structure + UtilsFct.set_load_at_x_on_simply_supported_euler_beams(
                    euler_beam_structure, loc_vehicle[i], f)

            # calculate the response of the beam
            u_structure, v_structure, a_structure = solver.calculate(euler_beam_structure.M_global,
                                                                     euler_beam_structure.C_global,
                                                                     euler_beam_structure.K_global, F_at_structure, dt,
                                                                     t, np.copy(u_structure), np.copy(v_structure),
                                                                     np.copy(a_structure))

            # update vehicle location
            loc_vehicle = loc_vehicle + velocity * dt

            # store displacement results
            all_u_beam.append(u_structure.tolist())

        # load expected results
        expected_results = json.load(open('tests/test_data/expected_data_test_two_wagons.json', "r"))

        # Assert results
        np.testing.assert_almost_equal(all_u_beam, expected_results["u_beam"])

        if INSPECT_RESULTS:
            all_u_beam = np.array(all_u_beam)

            fig, ax = plt.subplots(figsize=(6, 5))
            # plot displacement results of the centre of the beam and the bogie
            ax.plot(time[:-1], all_u_beam[:, len(u_structure) // 2 + 1], color='r', label="numerical")
            ax.set_ylabel("Displacement beam [m]")
            ax.set_xlabel("Time [s]")
            ax.grid()
            ax.legend()
            plt.tight_layout()
            plt.show()
