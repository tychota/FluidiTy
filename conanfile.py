from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class Fluidity(ConanFile):
    settings = ("os", "compiler", "build_type", "arch")
    test_requires = "gtest/1.17.0"

    def generate(self):
        CMakeDeps(self).generate()
        toolchain = CMakeToolchain(self, generator="Ninja")
        toolchain.user_presets_path = False
        toolchain.generate()
