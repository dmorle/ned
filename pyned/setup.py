import os
import configparser
from setuptools import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib

config = configparser.RawConfigParser()
with open("${CMAKE_CURRENT_SOURCE_DIR}/pyned-env/pyvenv.cfg", "r") as f:
    config.read_string("[t]\n" + f.read())
config = config["t"]

inc_dirs = [
    "${CMAKE_CURRENT_SOURCE_DIR}/include/",
    "${CMAKE_SOURCE_DIR}/libned/include/",
    "${CMAKE_CURRENT_SOURCE_DIR}/pyned-env/Include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/pyned-env/include/"
    "%s/Include/" % config["home"],
    "%s/include/" % config["home"]
]
lib_dirs = [
    "%s/libs/" % config["home"],
    "${CMAKE_BINARY_DIR}/libned/"
]
libs = [
    "python38_d" if "${CMAKE_BUILD_TYPE}" == "Debug" else "python38",
    "libned"
]
core_src = [os.path.join(root, file)
            for root, dirs, files in os.walk("${CMAKE_CURRENT_SOURCE_DIR}/source/core/")
            for file in files
            if file.split(".")[-1] == "cpp"]
lang_src = [os.path.join(root, file)
            for root, dirs, files in os.walk("${CMAKE_CURRENT_SOURCE_DIR}/source/lang/")
            for file in files
            if file.split(".")[-1] == "cpp"]

core_ext = Extension(
    name="core",
    sources=core_src,
    include_dirs=inc_dirs,
    library_dirs=lib_dirs,
    libraries=libs,
    language="c++"
)

lang_ext = Extension(
    name="lang",
    sources=lang_src,
    include_dirs=inc_dirs,
    library_dirs=lib_dirs,
    libraries=libs,
    language="c++"
)

setup(
    name="pyned",
    version="0.0.1",
    description="Python interface for the NEtwork Description language",
    author="Dario Morle",
    ext_package="pyned.cpp",
    ext_modules=[core_ext, lang_ext],
    packages=["pyned"],
    package_dir={"pyned": "./pyned"}
)
