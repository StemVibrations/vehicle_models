import os
import sys
import site
import subprocess
from shutil import rmtree


class TestCopy():

    def test_install_package_ten_dof(self):
        """
        Test install of 10 DOF uvec
        """

        # uninstall the package
        subprocess.run(['pip', 'uninstall', '-y', "UVEC"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Install the package in normal mode
        subprocess.run(['pip', 'install', '.'])

        from UVEC import uvec_ten_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_ten_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(site.getsitepackages()[0], r"UVEC/uvec_ten_dof_vehicle_2D") == path_uvec

    def test_install_package_ten_dof_editable(self):
        """
        Test install of 10 DOF uvec in editable mode
        """

        # uninstall the package
        subprocess.run(['pip', 'uninstall', '-y', "UVEC"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Install the package in normal mode
        subprocess.run(['pip', 'install', '-e', '.'])

        from UVEC import uvec_ten_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_ten_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(os.getcwd(), r"UVEC/uvec_ten_dof_vehicle_2D") == path_uvec

    def test_install_package_two_dof(self):
        """
        Test install of 2 DOF uvec
        """

        # uninstall the package
        subprocess.run(['pip', 'uninstall', '-y', "UVEC"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Install the package in normal mode
        subprocess.run(['pip', 'install', '.'])

        from UVEC import uvec_two_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_two_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(site.getsitepackages()[0], r"UVEC/uvec_two_dof_vehicle_2D") == path_uvec

    def test_install_package_two_dof_editable(self):
        """
        Test install of 2 DOF uvec in editable mode
        """

        # uninstall the package
        subprocess.run(['pip', 'uninstall', '-y', "UVEC"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Install the package in normal mode
        subprocess.run(['pip', 'install', '-e', '.'])

        from UVEC import uvec_two_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_two_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(os.getcwd(), r"UVEC/uvec_two_dof_vehicle_2D") == path_uvec
