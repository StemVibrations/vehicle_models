import unittest
import json
from typing import Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from uvec_ten_dof_vehicle_2D.uvec import uvec
from uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit

from tests.utils import UtilsFct
from tests.analytical_solutions.moving_vehicle import TwoDofVehicle


INSPECT_RESULTS = False


class TestSpringDamperModel(unittest.TestCase):
    """
    Test class for the spring damper model
    """

    def inspect_results(self,
                        json_input_file: Dict[str, Any],
                        beam_parameters: Dict[str, Any],
                        time: npt.NDArray[np.float64],
                        all_u_beam: List[npt.NDArray[np.float64]],
                        all_u_bogie: List[npt.NDArray[np.float64]]):
        """
        Plots the results of the spring damper model

        Args:
            - json_input_file (Dict[str, Any]): json input file
            - beam_parameters (Dict[str, Any]): beam parameters
            - time (npt.NDArray[np.float64]): time array
            - all_u_beam (List[npt.NDArray[np.float64]]): list containing the beam displacements
            - all_u_bogie (List[npt.NDArray[np.float64]]): list containing the bogie displacements
        """

        all_u_beam = np.array(all_u_beam)
        all_u_bogie = np.array(all_u_bogie)

        fig, ax = plt.subplots(2, 1, figsize=(6, 9))
        # plot displacement results of the centre of the beam and the bogie
        ax[0].plot(time[:-1], -all_u_beam[:, len(all_u_beam[0]) // 2 + 1], color='r', label="numerical")
        ax[1].plot(time[:-1], -all_u_bogie, color='r', label="numerical")

        ss = TwoDofVehicle()
        ss.vehicle(json_input_file["parameters"]["bogie_mass"], json_input_file["parameters"]["wheel_mass"],
                   json_input_file["parameters"]["velocity"],
                   json_input_file["parameters"]["wheel_stiffness"], json_input_file["parameters"]["wheel_damping"])
        ss.beam(beam_parameters["E"], beam_parameters["I"], beam_parameters["rho"], beam_parameters["A"],
                beam_parameters["total_length_beam"])
        ss.compute()

        ax[0].plot(ss.time, ss.displacement[:, 0], color='b', linestyle="--", label="analytical")
        ax[1].plot(ss.time, ss.displacement[:, 1], color='b', linestyle="--", label="analytical")
        ax[0].set_ylabel("Displacement beam [m]")
        ax[1].set_ylabel("Displacement bogie [m]")
        ax[1].set_xlabel("Time [s]")
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    def run_spring_damper_model(self, json_input_file: Dict[str, Any], beam_parameters: Dict[str, Any],
                                time: npt.NDArray[np.float64]) \
            -> Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]]]:
        """
        Run the spring damper model on a beam

        Args:
            - json_input_file (Dict[str, Any]): json input file
            - beam_parameters (Dict[str, Any]): beam parameters
            - time (npt.NDArray[np.float64]): time array

        Returns:
            - Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]]]: tuple containing the beam and
            bogie displacements
        """

        # set vehicle location parameters
        loc_vehicle = 0.0

        velocity = json_input_file["parameters"]["velocity"]

        # Euler beam parameters
        n_beams =  beam_parameters["n_elements"]
        length_beam = beam_parameters["total_length_beam"]
        E = beam_parameters["E"]
        I = beam_parameters["I"]
        L = length_beam / n_beams
        rho = beam_parameters["rho"]
        A = beam_parameters["A"]
        omega_1 = beam_parameters["omega_1"]
        omega_2 = beam_parameters["omega_2"]

        # Create the euler beam structure
        euler_beam_structure = UtilsFct.create_simply_supported_euler_beams(n_beams, E, I, L, rho, A, omega_1, omega_2)

        u_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        v_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        a_structure = np.zeros(euler_beam_structure.K_global.shape[0])

        # initialize time integration solver for simply supported beam
        solver = NewmarkExplicit()

        # get dt
        dt = time[1] - time[0]
        json_input_file["dt"] = dt

        # initialize arrays to store results
        all_u_beam = []
        all_u_bogie = []

        # loop over time steps
        for t in range(len(time) - 1):

            # get vertical displacement at vehicle location on beam
            u_vert = UtilsFct.get_result_at_x_on_simply_supported_euler_beams(u_structure, euler_beam_structure,
                                                                              loc_vehicle)
            json_input_file["u"]["1"][1] = u_vert
            json_input_file["time_index"] = t

            # call uvec model and retrieve force at wheel, this is what is tested.
            return_json = uvec(json.dumps(json_input_file))
            json_input_file = json.loads(return_json)

            F_vehicle = [np.array(json_input_file["loads"]["1"][1])]

            # set force at vehicle location on beam
            F_at_structure = np.zeros(euler_beam_structure.K_global.shape[0])
            for f in F_vehicle:
                F_at_structure = F_at_structure + UtilsFct.set_load_at_x_on_simply_supported_euler_beams(
                    euler_beam_structure, loc_vehicle, f)

            # calculate the response of the beam
            u_structure, v_structure, a_structure = solver.calculate(euler_beam_structure.M_global,
                                                                     euler_beam_structure.C_global,
                                                                     euler_beam_structure.K_global,
                                                                     F_at_structure, dt, t, np.copy(u_structure),
                                                                     np.copy(v_structure), np.copy(a_structure))

            # update vehicle location
            loc_vehicle = loc_vehicle + velocity * dt

            # store displacement results
            all_u_beam.append(u_structure.tolist())
            all_u_bogie.append(json_input_file["state"]["u"][-3])

        return all_u_beam, all_u_bogie

    def test_spring_damper_model(self):
        """
        Tests a moving vehicle on a simply supported beam. Where the vehicle consists of a wheel which
        is in contact with the beam and a mass which is connected to the wheel with a spring and damper

        Based on Biggs "Introduction do Structural Dynamics", pp pg 322

        :return:
        """

        json_input_file = {"dt": 0.001,
                           "loads": {"1": [0, 0, 0]},
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
                                          "gravity_axis": 1,  # 0 = x, 1 = y, 2 = z
                                          "velocity": 100 / 3.6,
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

        beam_parameters = {"n_elements": 10,
                           "total_length_beam": 50,
                           "E": 2.87e9,
                           "I": 2.9,
                           "rho": 2303,
                           "A": 1,
                           "omega_1": 0,
                           "omega_2": 0
                           }

        time = np.linspace(0, 1.8, 2000)

        # run the spring damper model
        all_u_beam, all_u_bogie = self.run_spring_damper_model(json_input_file, beam_parameters, time)

        # inspect results if needed
        if INSPECT_RESULTS:
            self.inspect_results(json_input_file, beam_parameters, time, all_u_beam, all_u_bogie)

        # load expected results
        expected_results = json.load(open('tests/test_data/expected_data_test_spring_damper.json', "r"))

        # Assert results
        np.testing.assert_almost_equal(all_u_beam, expected_results["u_beam"])
        np.testing.assert_almost_equal(all_u_bogie, expected_results["u_bogie"])

    def test_spring_damper_model_with_irregularity(self):
        """
        Tests a moving vehicle on a simply supported beam with irregularities. Where the vehicle consists of a wheel
        which is in contact with the beam and a mass which is connected to the wheel with a spring and damper

        Based on Biggs "Introduction do Structural Dynamics", pp pg 322

        :return:
        """

        # set vehicle location parameters
        velocity = 100 / 3.6

        json_input_file = {"dt": 0.001,
                           "loads": {"1": [0, 0, 0]},
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
                                          "gravity_axis": 1,  # 0 = x, 1 = y, 2 = z
                                          "velocity": velocity,
                                          "wheel_configuration": [0],
                                          "irr_parameters": {"Av": 0.0002095}
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

        beam_parameters = {"n_elements": 10,
                           "total_length_beam": 50,
                           "E": 2.87e9,
                           "I": 2.9,
                           "rho": 2303,
                           "A": 1,
                           "omega_1": 0,
                           "omega_2": 0
                           }

        time = np.linspace(0, 1.8, 2000)

        # run the spring damper model
        all_u_beam, all_u_bogie = self.run_spring_damper_model(json_input_file, beam_parameters, time)

        if INSPECT_RESULTS:
            self.inspect_results(json_input_file, beam_parameters, time, all_u_beam, all_u_bogie)

        # load expected results
        expected_results = json.load(open('tests/test_data/expected_data_test_spring_damper_with_irregularities.json',
                                          "r"))

        # Assert results
        np.testing.assert_almost_equal(all_u_beam, expected_results["u_beam"])
        np.testing.assert_almost_equal(all_u_bogie, expected_results["u_bogie"])
