import os
import subprocess
import sys
import sysconfig


class TestCopy():

    def test_install_package(self):
        """
        Test install of 10 DOF uvec
        """

        # uninstall the package
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', "UVEC"])

        # Install the package in normal mode
        subprocess.run([sys.executable, '-m', 'pip', 'install', '.', '--force-reinstall'])

        # 10 DOF
        from UVEC import uvec_ten_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_ten_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(sysconfig.get_paths()["purelib"], "UVEC", "uvec_ten_dof_vehicle_2D") == path_uvec

        # 2 DOF
        from UVEC import uvec_two_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_two_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(sysconfig.get_paths()["purelib"], "UVEC", "uvec_two_dof_vehicle_2D") == path_uvec

    def test_install_package_editable(self):
        """
        Test install of 10 DOF uvec in editable mode
        """

        # uninstall the package
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', "UVEC"])

        # Install the package in normal mode
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--force-reinstall'])

        # 10 DOF
        from UVEC import uvec_ten_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_ten_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(os.getcwd(), "UVEC", "uvec_ten_dof_vehicle_2D") == path_uvec

        # 2 DOF
        from UVEC import uvec_two_dof_vehicle_2D as uvec
        assert uvec.UVEC_NAME == "uvec_two_dof_vehicle_2D"
        path_uvec = uvec.get_path_file(uvec.UVEC_NAME)
        assert os.path.join(os.getcwd(), "UVEC", "uvec_two_dof_vehicle_2D") == path_uvec
