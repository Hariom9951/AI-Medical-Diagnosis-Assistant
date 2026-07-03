from setuptools import find_packages, setup  # type: ignore
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Reads requirement lines from a file and filters install-trigger lines.

    Args:
        file_path (str): File path location of requirements.

    Returns:
        List[str]: List of requirements libraries.
    """
    requirements: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()

    # Clean comments and empty lines
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

    # Remove dynamic install-trigger mapping if present
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="ai-medical-diagnosis-assistant",
    version="0.1.0",
    author="Healthcare Engineering Team",
    author_email="engineering@medical-assistant.ai",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
