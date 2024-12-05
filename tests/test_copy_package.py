import os
import sysconfig
from shutil import rmtree

import uvec_ten_dof_vehicle_2D as uvec


class TestCopy():
    def test_copy_package(self):
        """
        Tests multiple moving vehicles on a simply supported beam.
        """
        site_packages_path = sysconfig.get_paths()["purelib"]
        egg_link_path = os.path.join(site_packages_path, f"{__name__.split('.')[0]}.egg-link")
        # if package installed as editable
        if os.path.isdir(egg_link_path):
            site_packages_path = egg_link_path

        uvec.set_path_file("test_folder")

        assert uvec.path_file == "test_folder"
        assert os.path.isdir("test_folder")
        assert os.path.isdir("test_folder/uvec_ten_dof_vehicle_2D")
        assert os.path.isdir("test_folder/uvec_two_dof_vehicle_2D")

        rmtree("test_folder")