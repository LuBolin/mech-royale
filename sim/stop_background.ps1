# Stops what sim\start_background.ps1 started: the training run, its worker processes and the dashboard server.
$all = Get-CimInstance Win32_Process
$roots = $all | Where-Object { $_.CommandLine -match 'evolve\.py|train_main\.cmd|serve_dashboard\.cmd|http\.server 8770' }
$ids = [System.Collections.Generic.HashSet[uint32]]::new()
$roots | ForEach-Object { [void]$ids.Add($_.ProcessId) }
do {  # add every descendant (venv launcher -> interpreter -> multiprocessing workers)
    $n = $ids.Count
    $all | Where-Object { $ids.Contains($_.ParentProcessId) } | ForEach-Object { [void]$ids.Add($_.ProcessId) }
} while ($ids.Count -gt $n)
foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue; "stopped $id" }
