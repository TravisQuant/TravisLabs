from setuptools import setup
from pybind11.setup_helpers import (
    Pybind11Extension,
    build_ext,
)


ext_modules = [
    Pybind11Extension(
        "travislabs_pricer",
        [
            "bindings.cpp",
        ],
        cxx_std=17,
    ),
]


setup(
    name="travislabs_pricer",
    version="0.1.0",
    description=(
        "TravisLabs C++ derivatives pricing engine"
    ),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": build_ext,
    },
)