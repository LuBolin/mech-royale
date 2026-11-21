# Starts training and the dashboard server as standalone Windows processes, so closing or
# restarting a terminal or Claude Code session does not stop them.
#   Start:  powershell -File sim\start_background.ps1
#   Stop:   powershell -File sim\stop_background.ps1
$root = Split-Path $PSScriptRoot -Parent
foreach ($cmd in @("train_main.cmd", "serve_dashboard.cmd")) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd /c `"$root\sim\$cmd`""; CurrentDirectory = $root }
    "started $cmd (pid $($r.ProcessId), code $($r.ReturnValue))"
}
