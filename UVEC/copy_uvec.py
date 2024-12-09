import os
import sys
import json
import inspect
import sysconfig
from shutil import copytree
from typing import Union

from .__version__ import __version__, __title__

PATH_FILE = None


def is_editable_install(folder_path: str) -> Union[bool, str]:
    """
    Check if the package is installed in editable mode.

    Args:
        - folder_path (str): Path to the folder where the editable package is installed.

    Returns:
        - Union[bool, str]: Path to the package is installed in editable mode, False otherwise.

    """
    if os.path.isdir(folder_path):
        with open(os.path.join(folder_path, "direct_url.json"), "r") as f:
            data = json.load(f)

            # Check if "url" exists and starts with "file://" (means that package is installed locally in -e mode)
            if "url" in data and data["url"].startswith("file://"):
                path_package = data["url"].split("file://")[1]

                with open(os.path.join(folder_path, "top_level.txt"), "r") as f:
                    packages = f.read().splitlines()

                return os.path.join(path_package, packages[0])
            else:
                # package installed in editable mode but not locally
                return False

    return False


def get_package_path() -> str:
    """
    Gets the path to the current package in the site-packages directory.

    Returns:
        - str: Path to the package.
    """

    package_name = "-".join([__title__, __version__])
    site_packages_path = sysconfig.get_paths()["purelib"]
    dist_info_path = os.path.join(site_packages_path, f"{package_name}.dist-info")

    editable_path = is_editable_install(dist_info_path)

    if not editable_path:
        # if installed in regular mode
        site_packages_path = sysconfig.get_paths()["purelib"]
        package_path = os.path.join(site_packages_path, __name__.split(".")[0])
        return package_path
    else:
        # if installed in editable mode
        return editable_path


def set_path_file(new_path: str, uvec_name: str):
    """
    Sets the global path_file variable and performs the copy operation.
    """
    global PATH_FILE
    PATH_FILE = new_path

    package_path = get_package_path()
    base_path = os.path.dirname(PATH_FILE)

    copytree(os.path.join(package_path, uvec_name), os.path.join(base_path, PATH_FILE, uvec_name), dirs_exist_ok=True)
