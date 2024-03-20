import unittest
import json

import matplotlib.pyplot as plt
import numpy as np

from uvec_ten_dof_vehicle_2D.uvec import uvec, uvec_static
from uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit

from tests.utils import UtilsFct
from tests.analytical_solutions.moving_vehicle import TwoDofVehicle


INSPECT_RESULTS = True


class TestStaticSpringDamperModel(unittest.TestCase):

    def test_static_spring_damper_model(self):
        """
        Tests a moving vehicle on a simply supported beam. Where the vehicle consists of a wheel which
        is in contact with the beam and a mass which is connected to the wheel with a spring and damper

        Based on Biggs "Introduction do Structural Dynamics", pp pg 322

        :return:
        """

        json_input_file = {                           "loads": {"1": [0, 0, 0]},
                           "parameters": {"n_carts": 1,
                                          "cart_inertia": 0,
                                          "cart_mass": 0,
                                          "cart_stiffness": 0,
                                          "cart_damping": 0,
                                          "bogie_distances": [0],
                                          "bogie_inertia": 0,
                                          "bogie_mass": 3000,
                                          "wheel_distances": [0],
                                          "wheel_mass": 5750,
                                          "wheel_stiffness": 1595e5,
                                          "wheel_damping": 1000,
                                          "contact_coefficient": 9.1e-8,
                                          "contact_power": 1,
                                          "gravity_axis": 1  # 0 = x, 1 = y, 2 = z
                                          },
                            "state": {"a": [],
                                      "u": [],
                                      "v": []
                                      },
                            "t": 0,
                            "theta": {"1": [0.0, 0.0, 0.0]},
                            "time_index": 0,
                            "u": {"1": [0.0, 0, 0.0]}
                            }

        # set vehicle location parameters
        loc_vehicle = 25
        velocity = 0.0

        # Euler beam parameters
        n_beams = 10
        length_beam = 50
        E = 2.87e9
        I = 2.9
        L = length_beam / n_beams
        rho = 2303
        A = 1
        omega_1 = 0
        omega_2 = 0

        # Create the euler beam structure
        euler_beam_structure = UtilsFct.create_simply_supported_euler_beams(n_beams, E, I, L, rho, A, omega_1,omega_2)

        u_structure = np.zeros(euler_beam_structure.K_global.shape[0])

        # set time integration parameters
        n_steps = 4
        time = np.linspace(0, 1, n_steps)
        dt = time[1] - time[0]

        json_input_file["dt"] = dt

        # initialize arrays to store results
        all_u_beam = []
        all_u_bogie = []

        # loop over time steps
        for t in range(n_steps-1):

            # get vertical displacement at vehicle location on beam
            u_vert = UtilsFct.get_result_at_x_on_simply_supported_euler_beams(u_structure, euler_beam_structure,
                                                                              loc_vehicle)
            json_input_file["u"]["1"][1] = u_vert
            json_input_file["time_index"] = t

            # call uvec model and retrieve force at wheel, this is what is tested.
            return_json = uvec_static(json.dumps(json_input_file))
            json_input_file = json.loads(return_json)

            F_vehicle = [np.array(json_input_file["loads"]["1"][1])]

            # set force at vehicle location on beam
            F_at_structure = np.zeros(euler_beam_structure.K_global.shape[0])
            for f in F_vehicle:
                F_at_structure = F_at_structure + UtilsFct.set_load_at_x_on_simply_supported_euler_beams(
                    euler_beam_structure, loc_vehicle, f)


            u_structure = np.linalg.solve(euler_beam_structure.K_global, F_at_structure)

            # update vehicle location
            loc_vehicle = loc_vehicle + velocity * dt

            # store displacement results
            all_u_beam.append(u_structure.tolist())
            all_u_bogie.append(json_input_file["state"]["u"][-3])

        # load expected results
        # expected_results = json.load(open('tests/test_data/expected_data_test_spring_damper.json',"r"))

        # expected beam deflection is constant over time and is the same as the deflection of a simply supported beam
        # with a point load in the middle.
        expected_beam_deflection = ((json_input_file["parameters"]["wheel_mass"] +
                                    json_input_file["parameters"]["bogie_mass"]) * 9.81 * length_beam**3/
                                    (48 * E * I))

        # Assert results
        np.testing.assert_allclose(expected_beam_deflection, -np.array(all_u_beam)[:, len(u_structure)//2-1])


        # the initial bogie displacement is calculated with a 0 beam deflection, thus it only takes into account
        # the vehicle stiffness and bogie mass. After the initial step, the following displacement is calculated
        # with the beam deflection and the vehicle stiffness.
        expected_diff_displacement_vehicle = (json_input_file["parameters"]["bogie_mass"] * 9.81 /
                                              json_input_file["parameters"]["wheel_stiffness"])

        final_bogie_displacement = expected_diff_displacement_vehicle + expected_beam_deflection
        expected_bogie_displacement = [expected_diff_displacement_vehicle,final_bogie_displacement,
                                       final_bogie_displacement ]

        np.testing.assert_almost_equal(expected_bogie_displacement, -np.array(all_u_bogie))


        if INSPECT_RESULTS:
            all_u_beam = np.array(all_u_beam)
            all_u_bogie = np.array(all_u_bogie)

            fig, ax = plt.subplots(2, 1, figsize=(6, 9))
            # plot displacement results of the centre of the beam and the bogie
            ax[0].plot(time[1:], -all_u_beam[:, len(u_structure)//2-1], color='r', label="numerical")
            ax[1].plot(time[1:], -all_u_bogie, color='r', label="numerical")

            # ss = TwoDofVehicle()
            # ss.vehicle(json_input_file["parameters"]["bogie_mass"], json_input_file["parameters"]["wheel_mass"], velocity,
            #            json_input_file["parameters"]["wheel_stiffness"], json_input_file["parameters"]["wheel_damping"])
            # ss.beam(E, I, rho, A, length_beam)
            # ss.compute()

            # ax[0].plot(ss.time, ss.displacement[:, 0], color='b', linestyle="--", label="analytical")
            # ax[1].plot(ss.time, ss.displacement[:, 1], color='b', linestyle="--", label="analytical")
            ax[0].set_ylabel("Displacement beam [m]")
            ax[1].set_ylabel("Displacement bogie [m]")
            ax[1].set_xlabel("Time [s]")
            ax[0].legend()
            ax[1].legend()
            plt.tight_layout()
            plt.show()

        a=1+1
        # # Assert results
        # np.testing.assert_almost_equal(all_u_beam, expected_results["u_beam"])
        # np.testing.assert_almost_equal(all_u_bogie, expected_results["u_bogie"])
