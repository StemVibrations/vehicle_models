import json

import numpy as np
import numpy.testing as npt

from UVEC.uvec_ten_dof_vehicle_2D.uvec import uvec

class TestNonLinearIteration:

    def test_spring_damper_model(self):
        """
        Test that the non-linear iteration in the uvec model works as expected. This is done by running two non-linear
        iterations at the same time step, and checking that the initial state remains the same, while the current state
        is updated.
        """

        input_dict = {
            "parameters": {
                "n_carts": 1,
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
                "contact_power": 3/2,
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
                "1": [0.0, 0.0, 0.0]
            },
            "time_index": 0,
            "u": {
                "1": [0.0, -10, 0.0]
            },
            "dt": 0.01
        }

        # run initialisation step
        return_json = uvec(json.dumps(input_dict))

        # change displacement and time and re-run uvec
        json_input_file = json.loads(return_json)
        json_input_file["u"]["1"][1] = -9.99
        json_input_file["time_index"] = 1
        json_input_file["t"] += json_input_file["dt"]
        return_json = uvec(json.dumps(json_input_file))
        output_dict_step_1_1 = json.loads(return_json)

        # change displacement but don't change the time and re-run uvec
        input_json_dict = json.loads(return_json)
        input_json_dict["u"]["1"][1] = -9.98

        return_json = uvec(json.dumps(input_json_dict))
        output_dict_step_1_2 = json.loads(return_json)

        # check that initial state remains but that current state is updated when doing a non-linear iteration
        npt.assert_array_almost_equal(output_dict_step_1_1["state"]["a_ini"], output_dict_step_1_2["state"]["a_ini"], decimal=12)
        npt.assert_array_almost_equal(output_dict_step_1_1["state"]["v_ini"], output_dict_step_1_2["state"]["v_ini"], decimal=12)
        npt.assert_array_almost_equal(output_dict_step_1_1["state"]["u_ini"], output_dict_step_1_2["state"]["u_ini"], decimal=12)

        assert not np.allclose(output_dict_step_1_1["state"]["a"], output_dict_step_1_2["state"]["a"])
        assert not np.allclose(output_dict_step_1_1["state"]["v"], output_dict_step_1_2["state"]["v"])
        assert not np.allclose(output_dict_step_1_1["state"]["u"], output_dict_step_1_2["state"]["u"])

