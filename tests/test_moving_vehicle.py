import matplotlib.pyplot as plt
import numpy as np
import unittest

from spring_damper_model.uvec import uvec
from spring_damper_model.newmark_solver import NewmarkExplicit
from tests.utils import TestUtils

class TestSpringDamperModel(unittest.TestCase):

    def test_spring_damper_model(self):
        """
        Tests a moving vehicle on a simply supported beam. Where the vehicle consists of a wheel which
        is in contact with the beam and a mass which is connected to the wheel with a spring and damper

        Based on Biggs "Introduction do Structural Dynamics", pp pg 322

        :return:
        """

        solver = NewmarkExplicit()

        parameters = {
            "n_carts": 1,
            "cart_intertia": 0,
            "cart_mass": 0,
            "cart_stiffness": 0,

            "cart_damping": 0,
            "bogie_distances": [0],
            "bogie_intertia": 0,
            "bogie_mass": 3000,
            "wheel_distances": [0],
            "wheel_mass": 5750,
            "wheel_stiffness": 1595e5,
            "wheel_damping": 1000,

            "contact_coefficient": 9.1e-8,
            "contact_power": 1
        }

        state = {
            "u": [],
            "v": [],
            "a": []
        }

        velocity = 100 / 3.6


        n_beams = 10
        E = 2.87e9
        I=2.9
        L = 5
        rho = 2303
        A = 1
        omega_1 = 0
        omega_2 = 0

        euler_beam_structure = TestUtils.create_simply_supported_euler_beams(n_beams, E, I, L, rho, A, omega_1,omega_2)

        n_steps= 2000
        time = np.linspace(0, 1.8, n_steps)

        dt = time[1] - time[0]
        theta_ini = np.zeros(3)[None,:]

        loc_vehicle = 0.0


        u_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        v_structure = np.zeros(euler_beam_structure.K_global.shape[0])
        a_structure = np.zeros(euler_beam_structure.K_global.shape[0])

        all_u_beam = []
        all_u_bogie = []
        for t in range(n_steps-1):

            u_vert =TestUtils.get_result_at_x_on_simply_supported_euler_beams(u_structure, euler_beam_structure,
                                                                              loc_vehicle)
            u_at_wheel = [[0,u_vert,0]]

            F_vehicle = uvec(u_at_wheel, theta_ini, dt, t, state, parameters)

            F_at_structure = np.zeros(euler_beam_structure.K_global.shape[0])

            for f in F_vehicle:
                F_at_structure = F_at_structure + TestUtils.set_load_at_x_on_simply_supported_euler_beams(euler_beam_structure, loc_vehicle, f)

            u_structure, v_structure, a_structure = solver.calculate(euler_beam_structure.M_global,
                                                                     euler_beam_structure.C_global,
                                                                     euler_beam_structure.K_global,
                                                                     F_at_structure, dt, t, np.copy(u_structure), np.copy(v_structure),
                                                                     np.copy(a_structure))
            loc_vehicle = loc_vehicle + velocity * dt

            all_u_beam.append(u_structure)
            all_u_bogie.append(state["u"][-3])

        all_u_beam = np.array(all_u_beam)
        all_u_bogie = np.array(all_u_bogie)

        plt.plot(time[:-1], -all_u_beam[:,len(u_structure)//2+1], color='r', label="beam")
        plt.plot(time[:-1], -all_u_bogie, color='b', label="bogie")
        plt.show()


