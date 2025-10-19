from setuptools import setup
from os import getenv

ext_modules = []
cmdclass = {}

# Installing cpp_extnetions is conditional and guided by
# CAYLEYPY_BUILD_CPP_EXT env variable. If it is set to 1,
# pybind11 and torch are needed at build time.
CAYLEYPY_BUILD_CPP = getenv("CAYLEYPY_BUILD_CPP")
if CAYLEYPY_BUILD_CPP:
    try:
        import pybind11
        from torch.utils import cpp_extension

        ext_modules = [
            cpp_extension.CppExtension(
                "cayleypy._cpp_algo",
                [
                    "cayleypy/_cpp_algo/pybind11.cpp",
                    "cayleypy/_cpp_algo/random_walks.cpp",
                    "cayleypy/_cpp_algo/random_walks_torch.cpp",
                ],
                include_dirs=[pybind11.get_include()],
                language="c++",
                extra_compile_args={"cxx": ["-std=c++17"]},
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        print("Building CayleyPy with optional torch cpp extentions")
    except Exception as e:
        if CAYLEYPY_BUILD_CPP:
            raise Exception(
                f"""
                Got the following exeption during setting up cpp extentions:
                \n{e}.\n 
                CAYLEYPY_BUILD_CPP is set to {CAYLEYPY_BUILD_CPP}. 
                If you want to install CayleyPy without cpp extnetions, set it to 0.
                """
            )
        else:
            raise Exception(e)
else:
    print("Didn't find torch during build -- building CayleyPy without optional torch cpp extentions ")


if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
