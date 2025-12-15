import json

import pytest

from UVEC.uvec_ten_dof_vehicle_2D.uvec import uvec

class TestStaticUvec:

    def test_static_uvec_with_initialisation_steps(self):
        """
        Test a static vehicle with initialisation steps. The test checks that the loads are scaled correctly with the
        number of initialisation steps.
        """

        json_input_file = {
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
                "static_initialisation": True,
                "initialisation_steps": 10,
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
        output_dict = json.loads(uvec(json.dumps(json_input_file)))

        maximum_force = -(json_input_file["parameters"]["wheel_mass"]  +
                          json_input_file["parameters"]["bogie_mass"]) * 9.81

        # check load after step 1
        assert pytest.approx(maximum_force/10) == output_dict["loads"]["1"][1]

        # move one step forward and re-run uvec
        output_dict["time_index"] += 1
        output_dict["t"] += output_dict["dt"]
        output_dict = json.loads(uvec(json.dumps(output_dict)))
        assert pytest.approx(maximum_force/10 * 2) ==  output_dict["loads"]["1"][1]

        # re-run uvec without advancing time step
        output_dict = json.loads(uvec(json.dumps(output_dict)))
        assert pytest.approx(maximum_force/10 * 2) == output_dict["loads"]["1"][1]

        # set time index to 9 and re-run uvec
        output_dict["time_index"] = 9
        output_dict = json.loads(uvec(json.dumps(output_dict)))
        assert pytest.approx(maximum_force) == output_dict["loads"]["1"][1]

        # set time index to 100 and re-run uvec
        output_dict["time_index"] = 100
        output_dict = json.loads(uvec(json.dumps(output_dict)))
        assert pytest.approx(maximum_force) == output_dict["loads"]["1"][1]

