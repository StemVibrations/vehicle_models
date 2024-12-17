import os
import json
import site
import sysconfig

from .__version__ import __version__, __title__


def editable_path(folder_path: str) -> str:
    """
    Collect the package location when installed in editable mode.

    Args:
        - folder_path (str): Path to the system package folder

    Returns:
        - str: Path to the package installed in editable mode

    """
    with open(os.path.join(folder_path, "direct_url.json"), "r") as f:
        data = json.load(f)

    path_package = data["url"].split("file://")[1]

    with open(os.path.join(folder_path, "top_level.txt"), "r") as f:
        packages = f.read().splitlines()

    return os.path.join(path_package, packages[0])


def is_installed_editable(package_name: str) -> bool:
    """
    Checks if the given package is installed in editable mode.
    Returns True if installed in editable mode, otherwise False.

    Args:
        - package_name (str): Name of the package to check.

    Returns:
        - bool: True if installed in editable mode, otherwise False.
    """

    return os.path.isfile(os.path.join(site.getsitepackages()[0], f"__editable__.{package_name}.pth"))


def get_package_path() -> str:
    """
    Gets the path to the current package in the site-packages directory.

    Returns:
        - str: Path to the package.
    """

    package_name = "-".join([__title__, __version__])
    site_packages_path = sysconfig.get_paths()["purelib"]
    dist_info_path = os.path.join(site_packages_path, f"{package_name}.dist-info")

    is_editable = is_installed_editable(package_name)

    if not is_editable:
        # if installed in regular mode
        site_packages_path = sysconfig.get_paths()["purelib"]
        package_path = os.path.join(site_packages_path, __name__.split(".")[0])
        return package_path
    else:
        # if installed in editable mode
        return editable_path(dist_info_path)


def get_path_file(uvec_name: str) -> str:
    """
    Sets the global path_file variable and performs the copy operation.

    Args:
        - uvec_name (str): Name of the uvec file to copy.

    Returns:
        - str: Global path to the uvec file.
    """

    package_path = get_package_path()

    return os.path.join(package_path, uvec_name)
