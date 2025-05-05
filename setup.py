from setuptools import setup, find_packages


def load_requirements(file_path):
    requirements = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


setup(
    name="py_ttcross",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=load_requirements("requirements.txt"),
)
