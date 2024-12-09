import os
from shutil import rmtree


class TestCopy():
    def test_copy_package_ten_dof(self):
        """
        Test copy the 10 DOF uvec
        """

        from UVEC import uvec_ten_dof_vehicle_2D as uvec
        uvec.set_path_file("test_folder", uvec.UVEC_NAME)

        assert uvec.UVEC_NAME == "uvec_ten_dof_vehicle_2D"
        assert os.path.isdir("test_folder")
        assert os.path.isdir("test_folder/uvec_ten_dof_vehicle_2D")
        rmtree("test_folder")

    def test_copy_package_two_dof(self):
        """
        Test copy the 2 DOF uvec
        """

        from UVEC import uvec_two_dof_vehicle_2D as uvec
        uvec.set_path_file("test_folder", uvec.UVEC_NAME)

        assert uvec.UVEC_NAME == "uvec_two_dof_vehicle_2D"
        assert os.path.isdir("test_folder")
        assert os.path.isdir("test_folder/uvec_two_dof_vehicle_2D")
        rmtree("test_folder")