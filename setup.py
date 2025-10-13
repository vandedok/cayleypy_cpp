# from glob import glob
# from setuptools import setup
# from pybind11.setup_helpers import Pybind11Extension

# ext_modules = [
#     Pybind11Extension(
#         "python_example",
#         sorted(glob("cayleypy/*.cpp")),  # Sort source files for reproducibility
#     ),
# ]

# setup(..., ext_modules=ext_modules)
from setuptools import setup, Extension
import pybind11


if __name__ == "__main__":
    setup(
        name="cayleypy", 
        ext_modules=[
            Extension(
                "cayleypy._cpp_algo",
                ["cayleypy/_cpp_algo/random_walks.cpp"],
                include_dirs=[pybind11.get_include()],
                language="c++"
            ),
        ]
    )