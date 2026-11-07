# InspectAI Windows PowerShell Launcher
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$pythonExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

Write-Host "Starting InspectAI with $pythonExe..." -ForegroundColor Cyan

$argsList = @("run.py", "--host", $HostAddress, "--port", $Port)
if ($Reload) {
    $argsList += "--reload"
}

& $pythonExe $argsList

