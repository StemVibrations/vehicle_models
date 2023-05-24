
import numpy as np


class HertzianContact():

    def __init__(self):
        self.contact_coeff = None
        self.contact_power = None


    def calculate_contact_force(self, du):
        """
        Calculate contact force

        :param du: differential displacement
        :return:
        """

        contact_force = np.sign( -du) * np.nan_to_num(np.power( 1 /self.contact_coeff * -du, self.contact_power))
        return contact_force


    def calculate_contact_deformation(self, F):


        return np.sign(F) * self.contact_coeff * np.abs(F) ** (1 / self.contact_power)

