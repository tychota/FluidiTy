# The one-command GPU run for Windows.
#
# What this proves. FluidiTy has GPU code that has only ever been compiled, never run (see docs/gpu-runs.md). This
# script builds everything for a Windows machine with an NVIDIA GPU, runs every test that needs a GPU, runs them a
# second time, and runs each of them under NVIDIA's memory checker. It also runs the CPU tests, which is the first time
# they run on Windows with the MSVC compiler.
#
# The three test labels. `cpu` tests run everywhere, so this script runs them. `no_gpu` tests assert that the
# machine has NO usable GPU, so they run only in the CUDA container (a machine without a GPU) and would fail here:
# this script never runs them. `gpu` tests run only on hardware, which is what this script is for.
#
# How to run it. From the repository root, in PowerShell 7 (the command is `pwsh`; the older "Windows PowerShell 5"
# that ships with Windows is not enough, and the script says so and stops):
#
#     pwsh -NoProfile -ExecutionPolicy Bypass -File tools/run_gpu_tests.ps1
#
# (`-ExecutionPolicy Bypass` lets a script that was downloaded or copied run without changing any setting.)
#
# You need on PATH: git, uv, nvcc and nvidia-smi (CUDA toolkit and driver), and Visual Studio with the C++ tools.
# compute-sanitizer comes with the CUDA toolkit. CUDA has no 32-bit host support, so `cl` must be the x64 one. If
# the `cl` on PATH is not x64 (or there is none), the script finds Visual Studio itself (with vswhere) and loads
# its x64 build environment; or start from the "x64 Native Tools Command Prompt" and run pwsh there. Allow a long
# first run: Conan builds the packages.
#
# What to paste back. The last thing printed is a block between two lines of three backticks. Paste that whole block
# and nothing else is needed. The full output of every command is in out/gpu-run/<UTC time>/, one file per command;
# the script prints that folder before the block. Keep it if any line says FAIL.
#
# How to read the block (every line is printed, even when the run stops early):
#   commit:     the full hash, and `clean` or `modified` (any uncommitted change, untracked files included).
#               A modified tree never gives result PASS, because the run would not describe a commit.
#   gpu:        the GPU the tests use, its driver version and its index: CUDA_DEVICE_ORDER is set to PCI_BUS_ID, so
#               CUDA's device 0 is the GPU nvidia-smi lists as index 0 (or the first index of CUDA_VISIBLE_DEVICES).
#   toolkit:    the release line of `nvcc --version`. An nvcc older than 12.8 cannot build for sm_120 (the RTX 50
#               cards), so it reads `FAIL nvcc <version> is older than 12.8` and the build does not run.
#   build:      Conan install, CMake configure and CMake build of the preset windows-release-cuda.
#   cpu tests:  the tests labelled cpu, as <passed>/<total>. When the CUDA runtime DLL check below fails, no test
#               runs and this line reads `FAIL cudart64_*.dll is in no folder of PATH`.
#   gpu tests:  the tests labelled gpu, as <passed>/<total>. A skipped or disabled test is not a pass: the line
#               then reads `<passed>/<total> FAIL <n> skipped or disabled`.
#   repeat run: the same gpu tests run a second time, and PASS means the GPU wrote the same bytes twice: the same
#               tests, by name, with the same outcomes, and identical result digests. For each run the script sets
#               FLUIDITY_RESULT_DIGEST_FILE to its own file in the log folder; the gpu tests of the buffers, the sums
#               and the stable order append a named digest of every GPU result they check (WriteResultDigest in
#               src/fluidity/base/testing/base_support.h), and the two files must be equal and not empty.
#               The tests also append notes, findings that do not fail a test, and timings to notes-first.txt and
#               notes-second.txt in the same folder (FLUIDITY_RESULT_NOTES_FILE, WriteResultNote, WriteResultTiming);
#               notes are read, never compared.
#   notes:      the lines of notes-first.txt, and among them the discrete mismatches the gpu tests counted on scenes
#               with a turning or spinning solid. Never decides the result; any nonzero count means: send
#               notes-first.txt back with the block (docs/gpu-runs.md).
#   sanitizer:  each gpu test program under `compute-sanitizer --tool memcheck`, errors added up. It reads
#               `PASS 0 errors`, `FAIL <n> errors in <k> executables`, `FAIL not installed`, or one of two other
#               `FAIL` reasons (no gpu test programs found; errors not counted but a program failed).
#   result:     PASS only when every line above says PASS and the commit is clean.
#
# Before the tests run, the script checks that the CUDA runtime DLL (cudart64_*.dll) is in a folder of PATH, where the
# test programs look for it; when it is not, it prints its own FAIL line, puts the reason on the `cpu tests:` line
# of the block (the first line that could not run), and runs no test.
#
# A step that fails stops the steps that need it. Those lines read `FAIL not run`, and the info lines (commit, gpu,
# toolkit) read `NOT RUN` when their tool is missing, so the block always keeps its shape. A missing tool prints
# `FAIL <tool> not found on PATH`. Exit code: 0 when result is PASS, 1 otherwise.
#
# How this script was checked: parsed with PowerShell 7.4 in a Linux container and exercised there with stand-in
# tools; never run on Windows, never run against a GPU. The owner's run is its first real one.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Every native exit code is checked by hand below; no command may turn into an exception because of this setting.
$PSNativeCommandUseErrorActionPreference = $false

# Windows PowerShell 5 would fail later with an unreadable error, so say it plainly first. This check uses only
# syntax that version 5 parses.
if ($PSVersionTable.PSVersion.Major -lt 7) {
  Write-Host ('FAIL PowerShell 7 is required (run it as: ' +
    'pwsh -NoProfile -ExecutionPolicy Bypass -File tools/run_gpu_tests.ps1); this is version ' +
    $PSVersionTable.PSVersion.ToString())
  exit 1
}

$script:Preset = 'windows-release-cuda'
# CUDA counts GPUs fastest first by default; PCI_BUS_ID makes its device 0 the GPU nvidia-smi lists as index 0.
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
# The oldest nvcc that compiles for sm_120 (RTX 50 cards), which the CUDA presets build for.
$script:MinimumNvccVersion = [version]'12.8'
$script:RepoRoot = Split-Path -Parent $PSScriptRoot
$script:LogDirectory = ''
$script:LogCount = 0

# What the block prints, in order. Each line has its text and whether it counts as passed. Everything starts as not
# run, so an early stop or an unexpected error still prints a complete block.
$script:Report = [ordered]@{
  commit    = [pscustomobject]@{ Label = 'commit:'; Text = 'NOT RUN'; Passed = $false }
  gpu       = [pscustomobject]@{ Label = 'gpu:'; Text = 'NOT RUN'; Passed = $false }
  toolkit   = [pscustomobject]@{ Label = 'toolkit:'; Text = 'NOT RUN'; Passed = $false }
  build     = [pscustomobject]@{ Label = 'build:'; Text = 'FAIL not run'; Passed = $false }
  cpuTests  = [pscustomobject]@{ Label = 'cpu tests:'; Text = 'FAIL not run'; Passed = $false }
  gpuTests  = [pscustomobject]@{ Label = 'gpu tests:'; Text = 'FAIL not run'; Passed = $false }
  repeat    = [pscustomobject]@{ Label = 'repeat run:'; Text = 'FAIL not run'; Passed = $false }
  # Information only: it never makes the result FAIL.
  notes     = [pscustomobject]@{ Label = 'notes:'; Text = 'NOT RUN'; Passed = $true }
  sanitizer = [pscustomobject]@{ Label = 'sanitizer:'; Text = 'FAIL not run'; Passed = $false }
}

# Reporting -------------------------------------------------------------------------------------------------------

function Set-ReportLine {
  param([string]$Key, [string]$Text, [bool]$Passed)
  $script:Report[$Key].Text = $Text
  $script:Report[$Key].Passed = $Passed
}

function Write-StepLine {
  param([string]$Message)
  Write-Host ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message)
}

function Write-FailureLine {
  param([string]$Message)
  Write-Host ('FAIL ' + $Message)
}

# Prints the block the owner pastes back, saves a copy in the log folder, and returns whether result is PASS.
function Write-FinalBlock {
  $allPassed = $true
  foreach ($line in $script:Report.Values) {
    if (-not $line.Passed) {
      $allPassed = $false
    }
  }
  $resultText = 'FAIL'
  if ($allPassed) {
    $resultText = 'PASS'
  }
  $blockLines = @('FluidiTy GPU run')
  foreach ($line in $script:Report.Values) {
    $blockLines += ('{0} {1}' -f $line.Label.PadRight(11), $line.Text)
  }
  $blockLines += ('{0} {1}' -f 'result:'.PadRight(11), $resultText)
  if ($script:LogDirectory -ne '') {
    Set-Content -LiteralPath (Join-Path $script:LogDirectory 'summary.txt') -Value $blockLines -Encoding utf8
    Write-Host ('Full log: ' + $script:LogDirectory)
  }
  Write-Host '```text'
  foreach ($blockLine in $blockLines) {
    Write-Host $blockLine
  }
  Write-Host '```'
  return $allPassed
}

# Running commands ------------------------------------------------------------------------------------------------

function New-LogDirectory {
  $stamp = [DateTime]::UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'")
  $path = Join-Path (Join-Path (Join-Path $script:RepoRoot 'out') 'gpu-run') $stamp
  New-Item -ItemType Directory -Force -Path $path | Out-Null
  $script:LogDirectory = $path
}

# The path of a program found on PATH, or an empty string.
function Find-Tool {
  param([string]$Name)
  $found = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $found) {
    return ''
  }
  return $found.Source
}

# Runs one program, writes its command line and output to its own file in the log folder, and returns
# [pscustomobject]@{ ExitCode; Lines }. Lines is always an array (zero, one or many lines). The program is called
# with the call operator, so a path with spaces works. With -StdoutOnly, Lines holds standard output alone and
# standard error goes to the log file, for the commands whose output the script reads as data.
function Invoke-LoggedCommand {
  param(
    [string]$LogName,
    [string]$Executable,
    [string[]]$Arguments = @(),
    [string]$WorkingDirectory = '',
    [switch]$StdoutOnly
  )
  $script:LogCount += 1
  $logPath = Join-Path $script:LogDirectory ('{0:00}-{1}.log' -f $script:LogCount, $LogName)
  Set-Content -LiteralPath $logPath -Value ('$ ' + $Executable + ' ' + ($Arguments -join ' ')) -Encoding utf8
  if ($WorkingDirectory -eq '') {
    $WorkingDirectory = $script:RepoRoot
  }
  $global:LASTEXITCODE = 0
  $savedPreference = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  Push-Location -LiteralPath $WorkingDirectory
  try {
    if ($StdoutOnly) {
      $lines = @(& $Executable @Arguments 2>> $logPath | ForEach-Object { $_.ToString() })
    } else {
      $lines = @(& $Executable @Arguments 2>&1 | ForEach-Object { $_.ToString() })
    }
    $exitCode = $global:LASTEXITCODE
  } finally {
    Pop-Location
    $ErrorActionPreference = $savedPreference
  }
  if ($lines.Count -gt 0) {
    Add-Content -LiteralPath $logPath -Value $lines -Encoding utf8
  }
  Add-Content -LiteralPath $logPath -Value ('exit code: ' + $exitCode) -Encoding utf8
  return [pscustomobject]@{ ExitCode = $exitCode; Lines = $lines; LogPath = $logPath }
}

# Loads the Visual Studio x64 build environment into this process, so `cl` is on PATH. Returns an empty string on
# success, otherwise the reason it failed.
function Import-VisualStudioEnvironment {
  $programFilesX86 = [Environment]::GetFolderPath('ProgramFilesX86')
  if ([string]::IsNullOrEmpty($programFilesX86)) {
    return 'the Program Files (x86) folder is unknown, so vswhere.exe could not be located'
  }
  $vswhere = Join-Path $programFilesX86 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path -LiteralPath $vswhere)) {
    return 'vswhere.exe was not found, so Visual Studio could not be located'
  }
  $where = Invoke-LoggedCommand -LogName 'vswhere' -Executable $vswhere -StdoutOnly -Arguments @(
    '-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
    '-property', 'installationPath')
  $installPath = ''
  foreach ($line in $where.Lines) {
    if ($line.Trim() -ne '') {
      $installPath = $line.Trim()
      break
    }
  }
  if ($installPath -eq '') {
    return 'vswhere found no Visual Studio with the C++ x64 tools'
  }
  $developerBatch = Join-Path $installPath 'Common7\Tools\VsDevCmd.bat'
  if (-not (Test-Path -LiteralPath $developerBatch)) {
    return ('VsDevCmd.bat was not found in ' + $installPath)
  }
  # A batch file cannot change this process, so a small wrapper runs it, then prints the environment it set up
  # with `set`; the lines NAME=VALUE are copied into this process. The environment is not logged.
  $wrapperPath = Join-Path $script:LogDirectory 'load-vs-environment.cmd'
  # VsDevCmd.bat can return a nonzero code and still have set up the tools, so that code is printed (and logged)
  # instead of stopping; the caller checks whether an x64 `cl` is on PATH afterwards. chcp makes the output UTF-8.
  $wrapperLines = @(
    '@echo off',
    'chcp 65001>nul',
    'set VSCMD_SKIP_SENDTELEMETRY=1',
    ('call "' + $developerBatch + '" -arch=amd64 -host_arch=amd64 -no_logo'),
    'echo FLUIDITY_VSDEVCMD_EXIT_CODE=%errorlevel%',
    'set')
  Set-Content -LiteralPath $wrapperPath -Value $wrapperLines -Encoding ascii
  try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
  } catch {
    Write-Host '  (could not switch the console to UTF-8; continuing)'
  }
  $global:LASTEXITCODE = 0
  $environmentLines = @(& $wrapperPath 2>&1 | ForEach-Object { $_.ToString() })
  $note = ''
  foreach ($line in $environmentLines) {
    $pair = [regex]::Match($line, '^([^=\s][^=]*)=(.*)$')
    if (-not $pair.Success) {
      continue
    }
    if ($pair.Groups[1].Value -eq 'FLUIDITY_VSDEVCMD_EXIT_CODE') {
      if ($pair.Groups[2].Value -ne '0') {
        $note = 'VsDevCmd.bat exited with ' + $pair.Groups[2].Value
      }
      continue
    }
    [Environment]::SetEnvironmentVariable($pair.Groups[1].Value, $pair.Groups[2].Value, 'Process')
  }
  Set-Content -LiteralPath (Join-Path $script:LogDirectory 'load-vs-environment.log') -Encoding utf8 `
    -Value @(('environment lines: ' + $environmentLines.Count), ('note: ' + $note))
  return $note
}

# True when the `cl` first on PATH builds for x64. CUDA has no 32-bit host support. The Visual Studio prompts set
# VSCMD_ARG_TGT_ARCH; the folder name of the compiler is the second way to tell.
function Test-X64Compiler {
  $clPath = Find-Tool -Name 'cl'
  if ($clPath -eq '') {
    return $false
  }
  if ($env:VSCMD_ARG_TGT_ARCH -eq 'x64') {
    return $true
  }
  return ($clPath -like '*\Hostx64\x64\*')
}

# Checks that the tools are on PATH. Each missing tool prints its own FAIL line. Returns the number missing.
function Test-RequiredTools {
  $missing = 0
  foreach ($name in @('git', 'uv', 'nvcc', 'nvidia-smi')) {
    if ((Find-Tool -Name $name) -eq '') {
      Write-FailureLine ($name + ' not found on PATH')
      $missing += 1
    }
  }
  if (-not (Test-X64Compiler)) {
    Write-StepLine 'no x64 cl on PATH; loading the Visual Studio x64 environment'
    $reason = Import-VisualStudioEnvironment
    if (-not (Test-X64Compiler)) {
      Write-FailureLine 'cl for x64 not found (start from "x64 Native Tools" or let the script find Visual Studio)'
      if ($reason -ne '') {
        Write-Host ('  ' + $reason)
      }
      $missing += 1
    }
  }
  return $missing
}

# Info lines ------------------------------------------------------------------------------------------------------

function Read-CommitLine {
  if ((Find-Tool -Name 'git') -eq '') {
    return
  }
  $hash = Invoke-LoggedCommand -LogName 'git-rev-parse' -Executable (Find-Tool -Name 'git') -StdoutOnly `
    -Arguments @('rev-parse', 'HEAD')
  $status = Invoke-LoggedCommand -LogName 'git-status' -Executable (Find-Tool -Name 'git') -StdoutOnly `
    -Arguments @('status', '--porcelain')
  if ($hash.ExitCode -ne 0 -or $status.ExitCode -ne 0 -or $hash.Lines.Count -lt 1) {
    Set-ReportLine -Key 'commit' -Text 'FAIL git could not read the commit' -Passed $false
    Write-FailureLine 'git could not read the commit (see the git-* logs)'
    return
  }
  $changedLines = @($status.Lines | Where-Object { $_.Trim() -ne '' })
  if ($changedLines.Count -gt 0) {
    Set-ReportLine -Key 'commit' -Text ($hash.Lines[0].Trim() + ' (modified)') -Passed $false
  } else {
    Set-ReportLine -Key 'commit' -Text ($hash.Lines[0].Trim() + ' (clean)') -Passed $true
  }
}

# The nvidia-smi index of the GPU the tests use: the first entry of CUDA_VISIBLE_DEVICES when it is set, else 0.
function Find-GpuIndex {
  if ($env:CUDA_VISIBLE_DEVICES) {
    return ($env:CUDA_VISIBLE_DEVICES -split ',')[0].Trim()
  }
  return '0'
}

function Read-GpuLine {
  $smi = Find-Tool -Name 'nvidia-smi'
  if ($smi -eq '') {
    return
  }
  $gpuIndex = Find-GpuIndex
  $query = Invoke-LoggedCommand -LogName 'nvidia-smi' -Executable $smi -StdoutOnly `
    -Arguments @('--query-gpu=name,driver_version', '--format=csv,noheader', ('--id=' + $gpuIndex))
  $firstGpu = @($query.Lines | Where-Object { $_.Trim() -ne '' })
  if ($query.ExitCode -ne 0 -or $firstGpu.Count -lt 1) {
    Set-ReportLine -Key 'gpu' -Text 'FAIL nvidia-smi did not list a GPU' -Passed $false
    Write-FailureLine 'nvidia-smi did not list a GPU (see the nvidia-smi log)'
    return
  }
  $parts = $firstGpu[0] -split ',', 2
  if ($parts.Count -lt 2) {
    Set-ReportLine -Key 'gpu' -Text ('FAIL unreadable nvidia-smi line: ' + $firstGpu[0]) -Passed $false
    return
  }
  Set-ReportLine -Key 'gpu' -Text ($parts[0].Trim() + ', driver ' + $parts[1].Trim() + ', index ' + $gpuIndex) `
    -Passed $true
  Write-StepLine ('the tests use GPU index ' + $gpuIndex + ' (' + $parts[0].Trim() + ')')
}

# Sets the toolkit line. Returns whether nvcc is at least $script:MinimumNvccVersion.
function Read-ToolkitLine {
  $nvcc = Find-Tool -Name 'nvcc'
  if ($nvcc -eq '') {
    return $false
  }
  $version = Invoke-LoggedCommand -LogName 'nvcc-version' -Executable $nvcc -StdoutOnly -Arguments @('--version')
  $versionLines = @($version.Lines | Where-Object { $_ -match 'release' })
  if ($version.ExitCode -ne 0 -or $versionLines.Count -lt 1) {
    Set-ReportLine -Key 'toolkit' -Text 'FAIL nvcc --version printed no release line' -Passed $false
    Write-FailureLine 'nvcc --version printed no release line (see the nvcc-version log)'
    return $false
  }
  $release = [regex]::Match($versionLines[0], 'release (\d+)\.(\d+)')
  if (-not $release.Success) {
    Set-ReportLine -Key 'toolkit' -Text ('FAIL unreadable nvcc release line: ' + $versionLines[0].Trim()) -Passed $false
    Write-FailureLine 'nvcc --version printed a release line without a version (see the nvcc-version log)'
    return $false
  }
  $nvccVersion = [version]($release.Groups[1].Value + '.' + $release.Groups[2].Value)
  if ($nvccVersion -lt $script:MinimumNvccVersion) {
    $text = 'FAIL nvcc {0} is older than {1}' -f $nvccVersion, $script:MinimumNvccVersion
    Set-ReportLine -Key 'toolkit' -Text $text -Passed $false
    Write-FailureLine ('nvcc {0} is older than {1}, which sm_120 (RTX 50 cards) needs' -f
      $nvccVersion, $script:MinimumNvccVersion)
    return $false
  }
  Set-ReportLine -Key 'toolkit' -Text $versionLines[0].Trim() -Passed $true
  return $true
}

# True when a folder of PATH holds the CUDA runtime DLL (cudart64_*.dll), which the test programs load at start: the
# shared runtime, linked by every target. When no folder does, it prints its own FAIL line and puts the reason on the
# `cpu tests:` line of the block, the first line that could not run.
function Test-CudaRuntimeDll {
  foreach ($folder in @($env:PATH -split [System.IO.Path]::PathSeparator)) {
    if ($folder.Trim() -eq '' -or -not (Test-Path -LiteralPath $folder -PathType Container)) {
      continue
    }
    $found = @(Get-ChildItem -LiteralPath $folder -Filter 'cudart64_*.dll' -File -ErrorAction SilentlyContinue)
    if ($found.Count -gt 0) {
      Write-StepLine ('CUDA runtime DLL: ' + $found[0].FullName)
      return $true
    }
  }
  Write-FailureLine ('the CUDA runtime DLL (cudart64_*.dll) is in no folder of PATH; add the folder of the toolkit ' +
    'that holds it (under CUDA_PATH, bin or bin\x64) to PATH')
  Set-ReportLine -Key 'cpuTests' -Text 'FAIL cudart64_*.dll is in no folder of PATH' -Passed $false
  return $false
}

# Build -----------------------------------------------------------------------------------------------------------

# Runs one `uv run --frozen <program> <arguments>` command. Returns whether it exited with 0, and prints a FAIL line
# when it did not.
function Invoke-UvStep {
  param([string]$Description, [string]$LogName, [string[]]$Arguments)
  Write-StepLine $Description
  $uv = Find-Tool -Name 'uv'
  $outcome = Invoke-LoggedCommand -LogName $LogName -Executable $uv -Arguments $Arguments
  if ($outcome.ExitCode -ne 0) {
    Write-FailureLine ('{0} exited with {1} (log: {2})' -f $Description, $outcome.ExitCode, $outcome.LogPath)
    return $false
  }
  return $true
}

# The same steps as the gate (tools/check_commit.py), for the Windows CUDA preset. Sets the build line.
function Invoke-Build {
  $env:CONAN_HOME = Join-Path (Join-Path $script:RepoRoot 'out') 'conan-home'
  $toolchainFolder = 'out/conan/' + $script:Preset
  $steps = @(
    @{ Name = 'uv sync'; Log = 'uv-sync'; Arguments = @('sync', '--frozen') },
    @{ Name = 'conan profile detect'; Log = 'conan-profile'; Arguments = @(
        'run', '--frozen', 'conan', 'profile', 'detect', '--exist-ok') },
    @{ Name = 'conan install'; Log = 'conan-install'; Arguments = @(
        'run', '--frozen', 'conan', 'install', '.', '--lockfile', 'conan.lock', '-of', $toolchainFolder,
        '-s', 'build_type=Release', '-s', 'compiler.cppstd=23', '-s:b', 'compiler.cppstd=23', '--build=missing') },
    @{ Name = 'cmake configure'; Log = 'cmake-configure'; Arguments = @(
        'run', '--frozen', 'cmake', '--preset', $script:Preset) },
    @{ Name = 'cmake build'; Log = 'cmake-build'; Arguments = @(
        'run', '--frozen', 'cmake', '--build', '--preset', $script:Preset) }
  )
  foreach ($step in $steps) {
    if (-not (Invoke-UvStep -Description $step.Name -LogName $step.Log -Arguments $step.Arguments)) {
      Set-ReportLine -Key 'build' -Text ('FAIL ' + $step.Name) -Passed $false
      return $false
    }
  }
  Set-ReportLine -Key 'build' -Text 'PASS' -Passed $true
  return $true
}

# Tests -----------------------------------------------------------------------------------------------------------

# Runs ctest on the preset with a label filter. Labels are regular expressions, so the filter is anchored: a bare
# `gpu` would also select the no_gpu tests. Returns ExitCode, Total and Passed (the counts ctest printed, or -1 when its
# summary line is missing), Skipped (how many tests were skipped or disabled), and Outcomes: a sorted array of
# `<test name> <outcome>` strings, one per test. ctest counts a skipped test as passed, so without the Skipped count a
# GPU test that quietly skipped itself would look like a pass.
#
# What ctest 4.4.4 (the version uv.lock pins) prints, captured from a throwaway project:
#   all pass:      `100% tests passed out of 3`                      (no "tests failed" part when none fails)
#   one fails:     `67% tests passed, 1 tests failed out of 3`, then `The following tests FAILED:` and
#                  `<tab>  2 - b (Failed)`
#   one skipped:   `2/3 Test #2: b ......***Skipped   0.00 sec`, `100% tests passed out of 3` (the skipped test
#                  counts as passed and is in the total), then `The following tests did not run:` and
#                  `<tab>  2 - b (Skipped)`
#   one disabled:  `2/3 Test #2: DISABLED_b ......***Not Run (Disabled)   0.00 sec`, `100% tests passed out of 2`
#                  (a disabled test is left out of the total), then `<tab>  2 - DISABLED_b (Disabled)`
#   no tests with --no-tests=error: `No tests were found!!!`, no summary line, and a non-zero exit code.
function Invoke-CtestRun {
  param([string]$LogName, [string]$LabelPattern)
  $uv = Find-Tool -Name 'uv'
  $run = Invoke-LoggedCommand -LogName $LogName -Executable $uv -Arguments @(
    'run', '--frozen', 'ctest', '--preset', $script:Preset, '-L', $LabelPattern, '--no-tests=error')
  $total = -1
  $failed = 0
  $outcomes = @()
  $skippedNames = @{}
  $disabledNames = @{}
  foreach ($line in $run.Lines) {
    $summary = [regex]::Match($line, '(\d+)% tests passed(?:, (\d+) tests? failed)? out of (\d+)')
    if ($summary.Success) {
      $total = [int]$summary.Groups[3].Value
      if ($summary.Groups[2].Success) {
        $failed = [int]$summary.Groups[2].Value
      }
    }
    $outcomePattern = '(\*\*\*\S+(?: Run)?(?: \(Disabled\))?|Passed)'
    $test = [regex]::Match($line, '^\s*\d+/\d+ Test\s+#\d+:\s+(\S+)\s+\.+\s*' + $outcomePattern)
    if ($test.Success) {
      $outcomes += ($test.Groups[1].Value + ' ' + $test.Groups[2].Value)
    }
    $notRun = [regex]::Match($line, '^\s*\d+ - (\S+) \((Skipped|Disabled)\)')
    if ($notRun.Success -and $notRun.Groups[2].Value -eq 'Skipped') {
      $skippedNames[$notRun.Groups[1].Value] = $true
    }
    if ($notRun.Success -and $notRun.Groups[2].Value -eq 'Disabled') {
      $disabledNames[$notRun.Groups[1].Value] = $true
    }
  }
  # A skipped test is inside the total and counted as passed, so it is taken off. A disabled test is outside the
  # total, so it is only reported.
  $passed = -1
  if ($total -ge 0) {
    $passed = $total - $failed - $skippedNames.Count
  }
  $sortedOutcomes = @($outcomes | Sort-Object)
  return [pscustomobject]@{
    ExitCode = $run.ExitCode
    Total = $total
    Passed = $passed
    Skipped = $skippedNames.Count + $disabledNames.Count
    Outcomes = $sortedOutcomes
    LogPath = $run.LogPath
  }
}

function Format-TestCounts {
  param($Run)
  if ($Run.Total -lt 0) {
    return '?/?'
  }
  return ('{0}/{1}' -f $Run.Passed, $Run.Total)
}

# The text after the counts on a failed tests line: `FAIL`, or `FAIL <n> skipped or disabled` when tests did not run.
function Format-TestVerdict {
  param($Run)
  if (Test-CtestRunPassed -Run $Run) {
    return 'PASS'
  }
  if ($Run.Skipped -gt 0) {
    return ('FAIL {0} skipped or disabled' -f $Run.Skipped)
  }
  return 'FAIL'
}

# A run passes when ctest exited with 0, every test passed, and none was skipped or disabled.
function Test-CtestRunPassed {
  param($Run)
  return ($Run.ExitCode -eq 0 -and $Run.Total -gt 0 -and $Run.Skipped -eq 0 -and $Run.Passed -eq $Run.Total)
}

# The cpu-labelled tests. The no_gpu tests are not run here: they assert that the machine has no usable GPU, so
# they would fail on this one (they run in the CUDA container).
function Invoke-CpuTests {
  Write-StepLine 'ctest, tests labelled cpu'
  $run = Invoke-CtestRun -LogName 'ctest-cpu' -LabelPattern '^cpu$'
  $isPass = Test-CtestRunPassed -Run $run
  Set-ReportLine -Key 'cpuTests' -Text ((Format-TestCounts -Run $run) + ' ' + (Format-TestVerdict -Run $run)) `
    -Passed $isPass
  if (-not $isPass) {
    Write-FailureLine ('cpu tests (log: ' + $run.LogPath + ')')
  }
}

# The digest file of one run of the gpu tests, in the log folder.
function Get-DigestFile {
  param([string]$RunName)
  return (Join-Path $script:LogDirectory ('digests-' + $RunName + '.txt'))
}

# The digest lines of a run, sorted (the tests append in the order ctest runs them), or an empty array.
function Read-DigestLines {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    return ,@()
  }
  return ,@(Get-Content -LiteralPath $Path | Where-Object { $_.Trim() -ne '' } | Sort-Object)
}

# The notes file of one run of the gpu tests, in the log folder: timings and findings that do not fail a test. They
# are kept apart from the digests and never compared; the owner sends them back (docs/gpu-runs.md).
function Get-NotesFile {
  param([string]$RunName)
  return (Join-Path $script:LogDirectory ('notes-' + $RunName + '.txt'))
}

# Sets the notes line from the first run's notes file: how many lines, and how many discrete mismatches.
function Set-NotesLine {
  $path = Get-NotesFile -RunName 'first'
  if (-not (Test-Path -LiteralPath $path)) {
    Set-ReportLine -Key 'notes' -Text 'none written' -Passed $true
    return
  }
  $lines = @(Get-Content -LiteralPath $path | Where-Object { $_.Trim() -ne '' })
  $mismatches = @($lines | Where-Object { $_.StartsWith('discrete mismatch') })
  $text = '{0} lines in notes-first.txt, {1} discrete mismatches' -f $lines.Count, $mismatches.Count
  if ($mismatches.Count -gt 0) {
    $text += ' (send notes-first.txt)'
  }
  Set-ReportLine -Key 'notes' -Text $text -Passed $true
}

# Runs the gpu-labelled tests with FLUIDITY_RESULT_DIGEST_FILE set to the run's digest file and
# FLUIDITY_RESULT_NOTES_FILE to its notes file, then unsets both.
function Invoke-GpuCtestRun {
  param([string]$LogName, [string]$RunName)
  $env:FLUIDITY_RESULT_DIGEST_FILE = Get-DigestFile -RunName $RunName
  $env:FLUIDITY_RESULT_NOTES_FILE = Get-NotesFile -RunName $RunName
  try {
    return (Invoke-CtestRun -LogName $LogName -LabelPattern '^gpu$')
  } finally {
    Remove-Item -Path Env:FLUIDITY_RESULT_DIGEST_FILE -ErrorAction SilentlyContinue
    Remove-Item -Path Env:FLUIDITY_RESULT_NOTES_FILE -ErrorAction SilentlyContinue
  }
}

# Runs the gpu-labelled tests. Returns the run, so the repeat run can compare with it.
function Invoke-GpuTests {
  Write-StepLine 'ctest, tests labelled gpu (first run)'
  $run = Invoke-GpuCtestRun -LogName 'ctest-gpu' -RunName 'first'
  $isPass = Test-CtestRunPassed -Run $run
  Set-ReportLine -Key 'gpuTests' -Text ((Format-TestCounts -Run $run) + ' ' + (Format-TestVerdict -Run $run)) `
    -Passed $isPass
  if (-not $isPass) {
    Write-FailureLine ('gpu tests (log: ' + $run.LogPath + ')')
  }
  return $run
}

# Runs the gpu-labelled tests again and requires the same tests with the same outcomes as the first run, and
# identical, non-empty result digests: the GPU wrote the same bytes both times.
function Invoke-RepeatRun {
  param($FirstRun)
  Write-StepLine 'ctest, tests labelled gpu (second run)'
  $second = Invoke-GpuCtestRun -LogName 'ctest-gpu-repeat' -RunName 'second'
  $sameCounts = ($second.Total -eq $FirstRun.Total -and $second.Passed -eq $FirstRun.Passed)
  $sameOutcomes = (($second.Outcomes -join "`n") -eq ($FirstRun.Outcomes -join "`n"))
  if (-not ((Test-CtestRunPassed -Run $second) -and $sameCounts -and $sameOutcomes)) {
    Set-ReportLine -Key 'repeat' -Text ('FAIL second run ' + (Format-TestCounts -Run $second)) -Passed $false
    Write-FailureLine ('repeat run differs from the first run or failed (log: ' + $second.LogPath + ')')
    return
  }
  $firstDigests = Read-DigestLines -Path (Get-DigestFile -RunName 'first')
  $secondDigests = Read-DigestLines -Path (Get-DigestFile -RunName 'second')
  if ($firstDigests.Count -eq 0) {
    Set-ReportLine -Key 'repeat' -Text 'FAIL no result digest was written' -Passed $false
    Write-FailureLine ('the gpu tests wrote no result digest (expected in ' + (Get-DigestFile -RunName 'first') + ')')
    return
  }
  if (($firstDigests -join "`n") -ne ($secondDigests -join "`n")) {
    Set-ReportLine -Key 'repeat' -Text 'FAIL result digests differ' -Passed $false
    Write-FailureLine ('the two runs wrote different result digests (compare digests-first.txt and ' +
      'digests-second.txt in ' + $script:LogDirectory + ')')
    return
  }
  Set-ReportLine -Key 'repeat' -Text ('PASS {0} identical digests' -f $firstDigests.Count) -Passed $true
}

# Sanitizer -------------------------------------------------------------------------------------------------------

# The path of compute-sanitizer. Order: the toolkit folder named by CUDA_PATH (compute-sanitizer\ first, then bin\),
# then PATH. The folder layout is recalled from CUDA installs, not verified here; PATH is the fallback if it is wrong.
# Empty when not found.
function Find-ComputeSanitizer {
  if ($env:CUDA_PATH) {
    $inToolkit = Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe'
    if (Test-Path -LiteralPath $inToolkit) {
      return $inToolkit
    }
  }
  $onPath = Find-Tool -Name 'compute-sanitizer'
  if ($onPath -ne '') {
    return $onPath
  }
  if ($env:CUDA_PATH) {
    $inBin = Join-Path $env:CUDA_PATH 'bin\compute-sanitizer.exe'
    if (Test-Path -LiteralPath $inBin) {
      return $inBin
    }
  }
  return ''
}

# The test programs behind the gpu-labelled tests, from ctest itself, so the list never goes stale. Returns an array
# of paths (empty when ctest lists none).
function Find-GpuTestPrograms {
  $buildFolder = Join-Path (Join-Path (Join-Path $script:RepoRoot 'out') 'build') $script:Preset
  $uv = Find-Tool -Name 'uv'
  $listing = Invoke-LoggedCommand -LogName 'ctest-list-gpu' -Executable $uv -StdoutOnly -Arguments @(
    'run', '--frozen', 'ctest', '--test-dir', $buildFolder, '-L', '^gpu$', '--show-only=json-v1')
  if ($listing.ExitCode -ne 0 -or $listing.Lines.Count -lt 1) {
    return ,@()
  }
  $programs = @()
  try {
    $parsed = ($listing.Lines -join "`n") | ConvertFrom-Json
    foreach ($test in @($parsed.tests)) {
      if ($null -ne $test.PSObject.Properties['command']) {
        $programs += [string]@($test.command)[0]
      }
    }
  } catch {
    return ,@()
  }
  # The unary comma keeps the result an array when it holds one path or none. The caller must not wrap the call in
  # @(): that would nest the array inside another one.
  return ,@($programs | Sort-Object -Unique)
}

# Runs every gpu test program under compute-sanitizer memcheck and sets the sanitizer line.
function Invoke-Sanitizer {
  $sanitizer = Find-ComputeSanitizer
  if ($sanitizer -eq '') {
    Set-ReportLine -Key 'sanitizer' -Text 'FAIL not installed' -Passed $false
    Write-FailureLine 'compute-sanitizer not found on PATH or in CUDA_PATH'
    return
  }
  $programs = Find-GpuTestPrograms
  if ($null -eq $programs -or @($programs).Count -lt 1) {
    Set-ReportLine -Key 'sanitizer' -Text 'FAIL no gpu test programs found' -Passed $false
    Write-FailureLine 'ctest listed no gpu test programs (see the ctest-list-gpu log)'
    return
  }
  $errorTotal = 0
  $failedPrograms = 0
  foreach ($program in @($programs)) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($program)
    Write-StepLine ('compute-sanitizer memcheck: ' + $name)
    $run = Invoke-LoggedCommand -LogName ('sanitizer-' + $name) -Executable $sanitizer `
      -WorkingDirectory (Split-Path -Parent $program) `
      -Arguments @('--tool', 'memcheck', '--error-exitcode', '1', $program)
    $errorsInProgram = 0
    foreach ($line in $run.Lines) {
      $summary = [regex]::Match($line, 'ERROR SUMMARY:\s*(\d+) errors?')
      if ($summary.Success) {
        $errorsInProgram += [int]$summary.Groups[1].Value
      }
    }
    $errorTotal += $errorsInProgram
    if ($run.ExitCode -ne 0 -or $errorsInProgram -gt 0) {
      $failedPrograms += 1
      Write-FailureLine ('sanitizer: {0} exited with {1}, {2} errors (log: {3})' -f
        $name, $run.ExitCode, $errorsInProgram, $run.LogPath)
    }
  }
  if ($failedPrograms -eq 0) {
    Set-ReportLine -Key 'sanitizer' -Text 'PASS 0 errors' -Passed $true
  } elseif ($errorTotal -eq 0) {
    Set-ReportLine -Key 'sanitizer' -Text ('FAIL 0 errors counted, {0} executables exited with failure' -f
      $failedPrograms) -Passed $false
  } else {
    Set-ReportLine -Key 'sanitizer' -Text ('FAIL {0} errors in {1} executables' -f $errorTotal, $failedPrograms) `
      -Passed $false
  }
}

# Main ------------------------------------------------------------------------------------------------------------

function Invoke-GpuRun {
  if (-not $IsWindows) {
    Write-FailureLine 'this script runs on Windows'
    return
  }
  New-LogDirectory
  Write-StepLine ('FluidiTy GPU run; logs in ' + $script:LogDirectory)
  Read-CommitLine
  $missingTools = Test-RequiredTools
  Read-GpuLine
  $toolkitIsRecentEnough = Read-ToolkitLine
  if ($missingTools -gt 0 -or -not $toolkitIsRecentEnough) {
    return
  }
  if (-not (Invoke-Build)) {
    return
  }
  if (-not (Test-CudaRuntimeDll)) {
    return
  }
  Invoke-CpuTests
  $firstGpuRun = Invoke-GpuTests
  Set-NotesLine
  if ($script:Report['gpuTests'].Passed) {
    Invoke-RepeatRun -FirstRun $firstGpuRun
  }
  Invoke-Sanitizer
}

$runPassed = $false
try {
  Invoke-GpuRun
} catch {
  Write-FailureLine ('the script stopped on an unexpected error: ' + $_.Exception.Message)
  Write-Host ('  at ' + $_.InvocationInfo.PositionMessage)
}
$runPassed = Write-FinalBlock
if ($runPassed) {
  exit 0
}
exit 1
