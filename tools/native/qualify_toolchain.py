"""Qualify the Windows x64 C++20 toolchain and optionally a real CUDA device."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PINNED_TOOLS = {"conan": "2.34.0", "cmake": "4.4.4", "ninja": "1.13.2"}
CRT_FLAGS = {"Debug": "MDd", "Release": "MD"}

HOST_CHECKS = r"""
#include <concepts>
#include <cstdio>

#if !defined(_MSC_VER) || !defined(_M_X64) || !defined(_MT) || !defined(_DLL)
#error The qualified host is MSVC x64 with the DLL CRT.
#endif
#if FLUIDITY_DEBUG_CRT != defined(_DEBUG)
#error The CRT debug mode does not match the build configuration.
#endif
static_assert(sizeof(void*) == 8);
static_assert(__cplusplus >= 202002L);

template <std::integral T>
constexpr T identity(T value) {
  return value;
}

static_assert(identity(7) == 7);

void print_host() {
#ifdef _DEBUG
  const char* crt = "MDd";
#else
  const char* crt = "MD";
#endif
  std::printf("architecture=x64 pointer_bits=%zu crt=%s msvc_ver=%d msvc_full_ver=%d "
              "cxx_standard=%ld\n", sizeof(void*) * 8, crt, _MSC_VER,
              _MSC_FULL_VER, __cplusplus);
}
"""

CPU_SOURCE = HOST_CHECKS + r"""
int main() {
  print_host();
}
"""

CUDA_SOURCE = HOST_CHECKS + r"""
#include <cuda_runtime.h>

__global__ void probe(int* result) {
  result[0] = 7;
#ifdef __CUDA_ARCH__
  result[1] = __CUDA_ARCH__ / 10;
#endif
}

bool check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    return false;
  }
  return true;
}

int main() {
  int device = 0;
  int runtime = 0;
  int driver = 0;
  cudaDeviceProp properties{};
  if (!check(cudaGetDevice(&device)) ||
      !check(cudaGetDeviceProperties(&properties, device)) ||
      !check(cudaRuntimeGetVersion(&runtime)) || !check(cudaDriverGetVersion(&driver))) {
    return 1;
  }

  int* result = nullptr;
  if (!check(cudaMallocManaged(&result, 2 * sizeof(int)))) {
    return 1;
  }
  probe<<<1, 1>>>(result);
  bool launched = check(cudaGetLastError()) && check(cudaDeviceSynchronize());
  if (launched) {
    print_host();
    std::printf("device_architecture=%d kernel_architecture=%d result=%d "
                "runtime_version=%d driver_api_version=%d\n",
                properties.major * 10 + properties.minor, result[1], result[0], runtime, driver);
    std::printf("%s\n", properties.name);
  }
  return check(cudaFree(result)) && launched ? 0 : 1;
}
"""


def resolve_tool(name):
    executable = shutil.which(name)
    return Path(executable).resolve() if executable else None


def parse_msvc_probe(probe_stdout, expected_crt):
    fields = dict(re.findall(r"(\w+)=([^\s]+)", probe_stdout))
    if fields.get("crt") != expected_crt:
        raise ValueError(f"expected {expected_crt} CRT, observed {fields.get('crt')!r}")
    if fields.get("architecture") != "x64" or fields.get("pointer_bits") != "64":
        raise ValueError(f"expected the MSVC x64 target with 64-bit pointers, got {probe_stdout!r}")
    if not re.fullmatch(r"\d{9}", fields.get("msvc_full_ver", "")):
        raise ValueError(f"missing full MSVC compiler version in probe output: {probe_stdout!r}")
    full_version_digits = fields["msvc_full_ver"]
    if fields.get("msvc_ver") != full_version_digits[:4] or int(fields.get("cxx_standard", "0")) < 202002:
        raise ValueError(f"expected consistent MSVC versions and C++20, got {probe_stdout!r}")
    return {
        "crt": expected_crt,
        "msvc_ver": int(fields["msvc_ver"]),
        "msvc_full_ver": int(full_version_digits),
        "compiler_version": f"{full_version_digits[:2]}.{full_version_digits[2:4]}.{full_version_digits[4:]}",
        "cxx_standard": int(fields["cxx_standard"]),
    }


def qualify(mode, cuda_architecture_code, probe_dir):
    tools = {}
    commands = []
    source_hashes = {}

    def tool(name):
        path = resolve_tool(name)
        if path is None:
            hint = " Use an x64 Visual Studio Native Tools shell." if name == "cl" else ""
            raise ValueError(f"required tool {name} was not found on PATH.{hint}")
        tools[name] = {"executable": str(path)}
        return path

    def run(*argv):
        argv = list(map(str, argv))
        result = subprocess.run(
            argv, cwd=probe_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", check=False,
        )
        commands.append({
            "argv": argv,
            "cwd": str(probe_dir),
            "exit_code": result.returncode,
            "output": result.stdout,
        })
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {argv}\n{result.stdout}")
        return result.stdout.strip()

    def version(name):
        output = run(tool(name), "--version")
        tools[name]["version"] = output
        return output

    def write_source(name, source):
        path = probe_dir / name
        contents = source.encode("utf-8")
        path.write_bytes(contents)
        source_hashes[name] = hashlib.sha256(contents).hexdigest()
        return path

    cl = tool("cl")
    if sys.version_info[:2] != (3, 13):
        raise ValueError(f"Python 3.13 is required, got {sys.version}")
    tools["python"] = {"executable": sys.executable, "version": sys.version}
    uv_version = re.match(r"uv (\d+)\.(\d+)\.(\d+)(?:\s|$)", version("uv"))
    if uv_version is None or tuple(map(int, uv_version.groups())) < (0, 13, 7):
        raise ValueError("uv >= 0.13.7 is required")
    for name, expected in PINNED_TOOLS.items():
        output = version(name)
        suffix = r"(?:\.git\.[\w.-]+)?" if name == "ninja" else ""
        observed = re.search(rf"(?<![\d.])(\d+\.\d+\.\d+){suffix}(?!\S)", output)
        if observed is None or observed.group(1) != expected:
            raise ValueError(f"{name} {expected} is required, got {tools[name]['version']!r}")

    cpu_probe_path = write_source("cpu probe.cc", CPU_SOURCE)
    host_configurations = {}
    for configuration, crt in CRT_FLAGS.items():
        probe_exe_path = probe_dir / f"{configuration}.exe"
        debug = int(configuration == "Debug")
        cl_output = run(
            cl, "/nologo", "/Bv", "/std:c++20", "/Zc:__cplusplus",
            f"/{crt}", f"/DFLUIDITY_DEBUG_CRT={debug}", f"/Fe{probe_exe_path}", cpu_probe_path,
        )
        host_configurations[configuration] = parse_msvc_probe(run(probe_exe_path), crt)
    host_version = host_configurations["Release"]["compiler_version"]
    if host_configurations["Debug"]["compiler_version"] != host_version:
        raise ValueError("Debug and Release used different MSVC versions")
    tools["cl"].update(version=host_version, version_output=cl_output)
    qualification_receipt = {
        "schema_version": 1,
        "mode": mode,
        "tools": tools,
        "host": {
            "architecture": "x64",
            "pointer_bits": 64,
            "visual_studio_version": os.environ.get("VSCMD_VER"),
            "toolset_version": os.environ.get("VCToolsVersion"),
            "windows_sdk_version": os.environ.get("WindowsSDKVersion", "").rstrip("\\"),
            "configurations": host_configurations,
        },
        "commands": commands,
        "source_sha256": source_hashes,
    }

    if mode == "cuda":
        if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
            raise ValueError("unset NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS before qualifying")
        version("nvcc")
        version("nvidia-smi")
        cuda_probe_path = write_source("device probe.cu", CUDA_SOURCE)
        cuda_configurations = {}
        for configuration, crt in CRT_FLAGS.items():
            probe_exe_path = probe_dir / f"{configuration} CUDA.exe"
            debug = int(configuration == "Debug")
            run(
                tools["nvcc"]["executable"], "-ccbin", cl, "--std=c++20", "--cudart=shared",
                f"--generate-code=arch=compute_{cuda_architecture_code},"
                f"code=[sm_{cuda_architecture_code},compute_{cuda_architecture_code}]",
                f"-Xcompiler=/{crt},/Zc:__cplusplus", f"-DFLUIDITY_DEBUG_CRT={debug}",
                "-o", probe_exe_path, cuda_probe_path,
            )
            host_line, device_line, device_name = run(probe_exe_path).splitlines()
            if parse_msvc_probe(host_line, crt) != host_configurations[configuration]:
                raise ValueError(f"CUDA {configuration} used a different host compiler configuration")
            fields = dict(re.findall(r"(\w+)=(\d+)", device_line))
            if (
                fields.get("kernel_architecture") != cuda_architecture_code
                or fields.get("device_architecture") != cuda_architecture_code
                or fields.get("result") != "7"
                or int(fields.get("runtime_version", "0")) <= 0
                or int(fields.get("driver_api_version", "0")) <= 0
                or not device_name
            ):
                raise ValueError(f"CUDA {configuration} probe did not match sm_{cuda_architecture_code}: {device_line}")
            cuda_configurations[configuration] = {
                "device": device_name,
                **{name: int(value) for name, value in fields.items()},
            }
        qualification_receipt["cuda"] = {
            "architecture": int(cuda_architecture_code),
            "runtime": "Shared",
            "host_compiler": str(cl),
            "configurations": cuda_configurations,
        }

    return qualification_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-architecture", default="120")
    args = parser.parse_args(argv)
    try:
        receipt_path = args.output.absolute()
        receipt_path.unlink(missing_ok=True)
        if not re.fullmatch(r"[1-9]\d{1,2}", args.cuda_architecture):
            raise ValueError("--cuda-architecture must be numeric, for example 120")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native probe ", dir=receipt_path.parent) as temp:
            probe_dir = Path(temp)
            qualification_receipt = qualify(args.mode, args.cuda_architecture, probe_dir)
            temp_receipt_path = probe_dir / "receipt.json"
            temp_receipt_path.write_text(json.dumps(qualification_receipt, indent=2) + "\n", encoding="utf-8")
            temp_receipt_path.replace(receipt_path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Toolchain qualification failed: {error}", file=sys.stderr)
        return 1
    print(f"Toolchain qualification succeeded: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
