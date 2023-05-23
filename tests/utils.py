
import numpy as np
from scipy.linalg import solve

class BeamElement:
    def __init__(self, E, I, L, rho, A, alpha, beta):
        self.E = E
        self.I = I
        self.L = L
        self.rho = rho
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.M, self.C, self.K = self.calculate_matrices()

    def calculate_matrices(self):
        K = (self.E * self.I / self.L ** 3) * np.array([[12, 6 * self.L, -12, 6 * self.L],
                                                        [6 * self.L, 4 * self.L ** 2, -6 * self.L, 2 * self.L ** 2],
                                                        [-12, -6 * self.L, 12, -6 * self.L],
                                                        [6 * self.L, 2 * self.L ** 2, -6 * self.L,
                                                         4 * self.L ** 2]])

        M = (self.rho * self.A * self.L / 420) * np.array([[156, 22 * self.L, 54, -13 * self.L],
                                                           [22 * self.L, 4 * self.L ** 2, 13 * self.L,
                                                            -3 * self.L ** 2],
                                                           [54, 13 * self.L, 156, -22 * self.L],
                                                           [-13 * self.L, -3 * self.L ** 2, -22 * self.L,
                                                            4 * self.L ** 2]])

        C = self.alpha * M + self.beta * K

        return M, C, K

    def calculate_disp_shape_functions(self,local_coordinate):

        l = self.L

        # precalculate for efficiency
        constant = 1 / 2
        x_l = local_coordinate / l
        x_l2 = x_l ** 2

        x2 = local_coordinate ** 2
        x3 = local_coordinate ** 3

        disp_shape_functions = np.zeros(4)
        disp_shape_functions[0] = constant * (1 + 2 * x_l ** 3 - 3 * x_l2)
        disp_shape_functions[1] = constant * (local_coordinate + (x3 / l ** 2) - 2 * (x2 / l))
        disp_shape_functions[2] = constant * (-2 * x_l ** 3 + 3 * x_l2)
        disp_shape_functions[3] = constant * ((x3 / l ** 2) - (x2 / l))

        return disp_shape_functions

class BeamStructure:
    def __init__(self, elements):
        self.elements = elements
        self.K_global, self.C_global, self.M_global = self.assemble_global_matrices()

    def assemble_global_matrices(self):
        n_elements = len(self.elements)

        K_global = np.zeros((2 * n_elements + 2, 2 * n_elements + 2))
        C_global = np.zeros((2 * n_elements + 2, 2 * n_elements + 2))
        M_global = np.zeros((2 * n_elements + 2, 2 * n_elements + 2))

        for i, element in enumerate(self.elements):
            K_global[2 * i:2 * i + 4, 2 * i:2 * i + 4] += element.K
            C_global[2 * i:2 * i + 4, 2 * i:2 * i + 4] += element.C
            M_global[2 * i:2 * i + 4, 2 * i:2 * i + 4] += element.M

        # impose boundary conditions
        K_global= np.delete(K_global, (0,-2), axis=0)
        K_global = np.delete(K_global, (0, -2), axis=1)

        C_global = np.delete(C_global, (0,-2), axis=0)
        C_global = np.delete(C_global, (0, -2), axis=1)

        M_global = np.delete(M_global, (0,-2), axis=0)
        M_global = np.delete(M_global, (0, -2), axis=1)


        return K_global, C_global, M_global

    def get_element_index_at_x(self, x):
        beam_lengths = [beam.L for beam in self.elements]

        node_coordinates = np.concatenate(([0], np.cumsum(beam_lengths)))

        # find the element that contains x
        element_index = np.where(node_coordinates > x)[0][0] - 1


        return element_index

    def get_local_coordinate_from_x(self, x):
        beam_lengths = [beam.L for beam in self.elements]

        node_coordinates = np.concatenate(([0], np.cumsum(beam_lengths)))

        # find the element that contains x
        element_index = np.where(node_coordinates > x)[0][0] - 1
        local_coordinate = x - node_coordinates[element_index]

        return local_coordinate, element_index

    def get_global_dof_indices(self, element_index):
        # get global dof indices
        global_dof_indices = [2 * element_index - 1, 2 * element_index, 2 * element_index + 1, 2 * element_index + 2]

        if element_index == 0:
            global_dof_indices[0] = None

        elif element_index == len(self.elements) - 1:
            last_index = global_dof_indices[-2]
            global_dof_indices[-2] = None
            global_dof_indices[-1] = last_index

        return global_dof_indices


class TestUtils():

    def __init__(self):
        pass

    @staticmethod
    def create_simply_supported_euler_beams( n_elements, E, I, L, rho, A, omega_1,omega_2):
        """
        Create a simply supported euler beam model

        :param n_elements: number of elements
        :param E: Young's modulus (N/mm^2)
        :param I: moment of inertia (m^4)
        :param L: length of beam (m)
        :param rho: mass density (kg/m^3)
        :param A: cross-sectional area (m^2)
        :param omega_1: damping coefficient1
        :param omega_2: damping coefficient2
        :return:

        """


        # initialize elements
        elements = [BeamElement(E, I, L, rho, A, omega_1, omega_2) for _ in range(n_elements)]

        # initialize structure
        return BeamStructure(elements)

    @staticmethod
    def get_result_at_x_on_simply_supported_euler_beams(u_vector, beam_structure, x):


        local_coordinate, element_index = beam_structure.get_local_coordinate_from_x(x)

        # get the element
        element = beam_structure.elements[element_index]
        global_dof_indices = beam_structure.get_global_dof_indices(element_index)
        disp_shape_functions = element.calculate_disp_shape_functions(local_coordinate)

        # get the displacement at x
        mask = global_dof_indices != np.array(None)
        disp = np.zeros(len(global_dof_indices))
        disp[mask] = u_vector[np.array(global_dof_indices)[mask].astype(int)]

        disp_at_x = np.sum(disp * disp_shape_functions)

        return disp_at_x

    @staticmethod
    def set_load_at_x_on_simply_supported_euler_beams(beam_structure, x, force):

        F_vector=np.zeros(beam_structure.K_global.shape[0])

        local_coordinate, element_index = beam_structure.get_local_coordinate_from_x(x)

        # get the element
        element = beam_structure.elements[element_index]
        global_dof_indices = beam_structure.get_global_dof_indices(element_index)
        disp_shape_functions = element.calculate_disp_shape_functions(local_coordinate)

        force_vector = disp_shape_functions * force

        mask = global_dof_indices != np.array(None)

        # add local force to global force vector
        F_vector[np.array(global_dof_indices)[mask].astype(int)] = force_vector[mask]

        return F_vector




if __name__ == '__main__':


    n_elements = 10
    E = 2.1e11
    I = 0.0001
    L = 1
    rho = 7850
    A = 0.01
    omega_1 = 0.1
    omega_2 = 0.2


    structure = TestUtils.create_simply_supported_euler_beams(n_elements, E, I, L, rho, A, omega_1, omega_2)

    u = np.zeros(2 * n_elements -2)
    u[1] = 1

    disp = TestUtils.get_result_at_x_on_simply_supported_euler_beams(u, structure,0.5)

    import matplotlib.pyplot as plt

    plt.plot(u[:,10])
    plt.show()

    print("Test passed!")