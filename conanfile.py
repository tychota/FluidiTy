from pathlib import Path

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class FluidityNative(ConanFile):
    settings = ("os", "compiler", "build_type", "arch")
    requires = "abseil/20250814.2", "flatbuffers/25.12.19", "blake3/1.8.5", "taywee-args/6.4.6"
    tool_requires = "flatbuffers/25.12.19"
    test_requires = "gtest/1.17.0"

    def generate(self):
        schema_tool = self.dependencies.build["flatbuffers"]
        executable_name = "flatc.exe" if self.settings_build.os == "Windows" else "flatc"
        flatc_executable = Path(schema_tool.package_folder, schema_tool.cpp_info.bindirs[0], executable_name)
        CMakeDeps(self).generate()
        toolchain = CMakeToolchain(self, generator="Ninja")
        toolchain.user_presets_path = False
        toolchain.variables["FLUIDITY_FLATC_EXECUTABLE"] = flatc_executable.as_posix()
        toolchain.generate()
