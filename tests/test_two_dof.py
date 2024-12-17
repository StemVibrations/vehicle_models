import json
import numpy as np

from UVEC.uvec_two_dof_vehicle_2D.uvec import uvec
from UVEC.uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit

from tests.utils import UtilsFct
from tests.analytical_solutions.moving_vehicle import TwoDofVehicle

INSPECT_RESULTS = False


class TestTwoDofNoContact():

    def test_two_dof_model(self):
        """
        Tests a moving vehicle on a simply supported beam. Where the vehicle consists of a wheel which
        is in contact with the beam and a mass which is connected to the wheel with a spring and damper

        Based on Biggs "Introduction do Structural Dynamics", pp pg 322

        :return:
        """

        json_input_file = {
            "dt": 0.001,
            "loads": {
                "1": [0, 0, 0]
            },
            "parameters": {
                "c": 1000,
                "k": 1595e5,
                "m1": 5720,
                "m2": 3000,
            },
            "state": {
                "a": 0,
                "u": 0,
                "u_beam": 0,
                "v": 0
            },
            "t": 0,
            "file_name": "test_two_dof.txt",
            "theta": {
                "1": [0.0, 0.0, 0.0]
            },
            "time_index": 0,
            "u": {
                "1": [0.0, 0, 0.0]
            }
        }

        # set vehicle location parameters
        loc_vehicle = 0.0
        velocity = 100 / 3.6

        # Euler beam parameters
        n_beams = 20
        length_beam = 50
        E = 2.87e9
        I = 2.9
        L = length_beam / n_beams
        rho = 2303
        A = 1
        omega_1 = 0
        omega_2 = 0

        # Create the euler beam structure
        euler_beam_structure = UtilsFct.create_simply_supported_euler_beams(n_beams, E, I, L, rho, A, omega_1, omega_2)

        u_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        v_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        a_structure = np.zeros(euler_beam_structure.K_global.shape[0])

        # initialize time integration solver for simply supported beam
        solver = NewmarkExplicit()

        # set time integration parameters
        n_steps = 10000
        time = np.linspace(0, 1.8, n_steps)
        dt = time[1] - time[0]

        # initialize arrays to store results
        u_beam = []
        uvec_mass = []
        uvec_displacement = []

        # loop over time steps
        for t in range(n_steps - 1):

            # get vertical displacement at vehicle location on beam
            u_vert = UtilsFct.get_result_at_x_on_simply_supported_euler_beams(u_structure, euler_beam_structure,
                                                                              [loc_vehicle])
            json_input_file["u"]["1"][1] = float(u_vert[0])

            # call uvec model and retrieve force at wheel, this is what is tested.
            return_json = uvec(json.dumps(json_input_file))
            json_input_file = json.loads(return_json)

            # get force at wheel
            F_vehicle = [np.array(json_input_file["loads"]["1"][1])]

            # set force at vehicle location on beam
            F_at_structure = np.zeros(euler_beam_structure.K_global.shape[0])
            for f in F_vehicle:
                F_at_structure = F_at_structure + UtilsFct.set_load_at_x_on_simply_supported_euler_beams(
                    euler_beam_structure, loc_vehicle, f)

            # calculate the response of the beam
            u_structure, v_structure, a_structure = solver.calculate(euler_beam_structure.M_global,
                                                                     euler_beam_structure.C_global,
                                                                     euler_beam_structure.K_global, F_at_structure, dt,
                                                                     t, np.copy(u_structure), np.copy(v_structure),
                                                                     np.copy(a_structure))

            # update vehicle location
            loc_vehicle = loc_vehicle + velocity * dt
            json_input_file["time_index"] = t

            # store displacement results
            u_beam.append(u_vert.item())
            uvec_mass.append(json_input_file["state"]["u"])
            uvec_displacement.append(json_input_file["state"]["u_beam"])

        # load expected results
        expected_results = json.load(open('tests/test_data/expected_data_two_dof.json', "r"))

        if INSPECT_RESULTS:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
            ax[0].plot(time[1:], np.array(uvec_displacement), label="beam", color="k")
            ax[0].plot(time[1:], np.array(u_beam), label="beam", color="r")
            ax[1].plot(time[1:], np.array(uvec_mass), label="beam", color="b")

            ss = TwoDofVehicle()
            ss.vehicle(json_input_file["parameters"]["m1"], json_input_file["parameters"]["m2"], velocity,
                       json_input_file["parameters"]["k"], json_input_file["parameters"]["c"])
            ss.beam(E, I, rho, A, length_beam)
            ss.compute()

            ax[0].plot(ss.time, ss.displacement[:, 0], color='r', marker="x", label="beam")
            ax[1].plot(ss.time, ss.displacement[:, 1], color='b', marker="x", label="vehicle")

            ax[0].set_ylabel("Beam [m]")
            ax[1].set_ylabel("Vehicle [m]")
            ax[1].set_xlabel("Time [s]")
            plt.tight_layout()
            plt.show()

        # Assert results
        np.testing.assert_almost_equal(u_beam, expected_results["u_beam"], decimal=2)
        np.testing.assert_almost_equal(uvec_mass, expected_results["u_bogie"])
