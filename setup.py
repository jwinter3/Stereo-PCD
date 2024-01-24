from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import shutil


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir="cpp", **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    def build_extensions(self):
        import subprocess

        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            subprocess.check_call(["cmake", ext.cmake_lists_dir], cwd=self.build_temp)
            subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)

            split_name = ext.name.split(".")

            shutil.copy(
                f"{self.build_temp}/module/{split_name[-1]}.so",
                f"{extdir}/{'/'.join(split_name[1:])}.so",
            )


setup(
    name="stereo_pcd",
    version="0.1",
    description="",
    url="",
    author="Jakub Winter",
    author_email="jakub.winter.stud@pw.edu.pl",
    license="MIT",
    packages=find_packages(),
    ext_modules=[CMakeExtension(name="stereo_pcd.stereo_match")],
    cmdclass={"build_ext": cmake_build_ext},
    zip_safe=False,
)
