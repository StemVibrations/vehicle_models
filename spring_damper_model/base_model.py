
import numpy as np

GRAVITY = 9.81

class Cart():
    def __init__(self):
        self.bogies = []
        self.dofs = []
        pass


class Bogie():
    def __init__(self):
        self.wheels = []
        self.dofs =[]
        pass

class Wheel():
    def __init__(self):
        self.dofs = []
        pass



class TrainModel():

    def __init__(self):

        self.n_carts = None
        self.n_bogies = None
        self.n_wheels = None


        self.cart_intertia = None
        self.cart_mass = None
        self.cart_stiffness = None
        self.cart_damping = None
        self.bogie_distances = None
        self.bogie_intertia = None
        self.bogie_mass = None
        self.wheel_intertia = None
        self.wheel_mass = None
        self.wheel_stiffness = None
        self.wheel_damping = None
        self.wheel_distances = None

        self.carts = []

        self.ndof = None
        self.contact_dofs = []


    def initialise(self):
        self.generate_components()


    def generate_components(self):

        for i in range(self.n_carts):
            cart = Cart()
            for j in range(self.n_bogies):
                bogie = Bogie()
                for k in range(self.n_wheels):
                    wheel = Wheel()
                    bogie.wheels.append(wheel)
                cart.bogies.append(bogie)
            self.carts.append(cart)

    def set_dofs(self):

        i_dof = 0

        for cart in self.carts:

            # cart has displacement and rotation dof
            cart.dofs = [i_dof, i_dof + 1]
            i_dof += 2

            for bogie in cart.bogies:

                    # bogie has displacement and rotation dof
                    bogie.dofs = [i_dof, i_dof + 1]
                    i_dof += 2

                    for wheel in bogie.wheels:

                        # wheel has displacement dof
                        wheel.dofs = [i_dof]
                        self.contact_dofs.append(i_dof)
                        i_dof += 1

        self.ndof = i_dof



    def generate_mass_matrix(self):
        """
        Generate the mass matrix
        """

        mass_matrix = np.zeros((self.ndof, self.ndof))


        for cart in self.carts:
            mass_matrix[cart.dofs[0], cart.dofs[0]] += self.cart_mass
            mass_matrix[cart.dofs[1], cart.dofs[1]] += self.cart_intertia

            for bogie in cart.bogies:
                mass_matrix[bogie.dofs[0], bogie.dofs[0]] += self.bogie_mass
                mass_matrix[bogie.dofs[1], bogie.dofs[1]] += self.bogie_intertia

                for wheel in bogie.wheels:
                    mass_matrix[wheel.dofs[0], wheel.dofs[0]] += self.wheel_mass

    def generate_stiffness_matrix(self):
        """
        Generate stiffness matrix
        """
        stiffness_matrix = np.zeros((self.ndof, self.ndof))

        for cart in self.carts:
            stiffness_matrix[cart.dofs[0], cart.dofs[0]] += self.cart_stiffness * len(cart.bogies)

            for b, bogie in enumerate(cart.bogies):
                stiffness_matrix[cart.dofs[1], cart.dofs[1]] += self.cart_stiffness * self.bogie_distances[b]**2

                stiffness_matrix[cart.dofs[0], bogie.dofs[0]] += -self.cart_stiffness
                stiffness_matrix[bogie.dofs[0], cart.dofs[1]] += -self.cart_stiffness

                stiffness_matrix[cart.dofs[1], bogie.dofs[0]] += self.cart_stiffness * self.bogie_distances[b]
                stiffness_matrix[bogie.dofs[0], cart.dofs[1]] += self.cart_stiffness * self.bogie_distances[b]

                stiffness_matrix[bogie.dofs[0], bogie.dofs[0]] += self.cart_stiffness


                stiffness_matrix[bogie.dofs[0], bogie.dofs[0]] += self.wheel_stiffness * len(bogie.wheels)
                for w, wheel in enumerate(bogie.wheels):
                    stiffness_matrix[bogie.dofs[1], bogie.dofs[1]] += self.wheel_stiffness * self.wheel_distances[w] ** 2
                    stiffness_matrix[bogie.dofs[0], wheel.dofs[0]] += -self.wheel_stiffness
                    stiffness_matrix[wheel.dofs[0], bogie.dofs[0]] += -self.wheel_stiffness

                    stiffness_matrix[bogie.dofs[1], wheel.dofs[0]] += self.wheel_stiffness * self.wheel_distances[w]
                    stiffness_matrix[wheel.dofs[0], bogie.dofs[1]] += self.wheel_stiffness * self.wheel_distances[w]

                    stiffness_matrix[wheel.dofs[0], wheel.dofs[0]] += self.wheel_stiffness

        return stiffness_matrix

    def generate_damping_matrix(self):
        """
        Generate damping matrix

        :return:
        """

        damping_matrix = np.zeros((self.ndof, self.ndof))

        for cart in self.carts:
            damping_matrix[cart.dofs[0], cart.dofs[0]] += self.cart_damping * len(cart.bogies)

            for b, bogie in enumerate(cart.bogies):
                damping_matrix[cart.dofs[1], cart.dofs[1]] += self.cart_damping * self.bogie_distances[b]**2

                damping_matrix[cart.dofs[0], bogie.dofs[0]] += -self.cart_damping
                damping_matrix[bogie.dofs[0], cart.dofs[0]] += -self.cart_damping

                damping_matrix[cart.dofs[1], bogie.dofs[0]] += self.cart_damping * self.bogie_distances[b]
                damping_matrix[bogie.dofs[0], cart.dofs[1]] += self.cart_damping * self.bogie_distances[b]

                damping_matrix[bogie.dofs[0], bogie.dofs[0]] += self.cart_damping

                damping_matrix[bogie.dofs[0], bogie.dofs[0]] += self.wheel_damping * len(bogie.wheels)
                for w, wheel in enumerate(bogie.wheels):
                    damping_matrix[bogie.dofs[1], bogie.dofs[1]] += self.wheel_damping * self.wheel_distances[w] ** 2
                    damping_matrix[bogie.dofs[0], wheel.dofs[0]] += -self.wheel_damping
                    damping_matrix[wheel.dofs[0], bogie.dofs[0]] += -self.wheel_damping

                    damping_matrix[bogie.dofs[1], wheel.dofs[0]] += self.wheel_damping * self.wheel_distances[w]
                    damping_matrix[wheel.dofs[0], bogie.dofs[1]] += self.wheel_damping * self.wheel_distances[w]

                    damping_matrix[wheel.dofs[0], wheel.dofs[0]] += self.wheel_damping

        return damping_matrix

    def generate_force_vector(self):
        """
        Generate force vector
        """
        force_vector = np.zeros(self.ndof)

        for cart in self.carts:
            force_vector[cart.dofs[0]] = -self.cart_mass * GRAVITY

            for bogie in cart.bogies:
                force_vector[bogie.dofs[0]] = - self.bogie_mass * GRAVITY

                for wheel in bogie.wheels:
                    force_vector[wheel.dofs[0]] = - self.wheel_mass * GRAVITY

        return force_vector
    def calculate_static_contact_force(self):
        """
        Calculate static contact force

        :return:
        """

        contact_forces=[]

        # Calculate static contact force
        for cart in self.carts:
            distributed_load = self.cart_mass * GRAVITY / len(cart.bogies)


            for bogie in cart.bogies:

                distributed_load = self.bogie_mass * GRAVITY / len(bogie.wheels) + distributed_load


                for wheel in bogie.wheels:
                    contact_force = self.wheel_mass * GRAVITY + distributed_load
                    contact_forces.append(contact_force)


        return np.array(contact_forces)


