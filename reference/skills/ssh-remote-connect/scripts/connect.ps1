[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $Command,

    [string] $Config = $env:SSH_REMOTE_CONFIG,
    [int] $Timeout = 60,
    [switch] $Raw,
    [switch] $Pty,
    [switch] $ShowConfig,
    [switch] $Interactive
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir "remote_exec.py"

$ArgsList = @()
if ($Config) {
    $ArgsList += @("--config", $Config)
}
$ArgsList += @("--timeout", [string]$Timeout)
if ($Raw) {
    $ArgsList += "--raw"
}
if ($Pty) {
    $ArgsList += "--pty"
}
if ($ShowConfig) {
    $ArgsList += "--show-config"
}
if ($Interactive) {
    $ArgsList += "--interactive"
}
if ($Command.Count -gt 0) {
    $ArgsList += $Command
}

python $Runner @ArgsList
exit $LASTEXITCODE
