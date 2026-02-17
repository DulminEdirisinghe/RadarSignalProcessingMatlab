# ================================
# run_all.ps1
# Step-by-step, automatic.
# Uses: wsl  (NOT wsl.exe)
# ================================

function Ask-YesNo($msg) {
    while ($true) {
        $ans = (Read-Host "$msg (y/n)").Trim().ToLower()
        if ($ans -match '^(y|n)$') { return ($ans -eq 'y') }
        Write-Host "Please type y or n."
    }
}

# ----------------
# CONFIG
# ----------------
$RadarNewDir = "C:\radar_receiver\radar_new"

$ReceiverScript = "C:\Users\asus\Documents\Projects\FYP\DSP\Matlab\RadarSignalProcessingMatlab\python_implementation\dataTransfer\radar_receiver.py"
$PlotScript     = "C:\Users\asus\Documents\Projects\FYP\DSP\Matlab\RadarSignalProcessingMatlab\python_implementation\processing\plottargets_opt.py"

$RemoteIP     = "192.168.33.180"
$RemoteHome   = "/home/root"
$RemoteFolder = "/mnt/ssd/test_capture_incoming"
$SshOpts      = "-oHostKeyAlgorithms=+ssh-rsa -oPubkeyAcceptedAlgorithms=+ssh-rsa"

# ----------------
# Validate scripts exist
# ----------------
if (!(Test-Path -LiteralPath $ReceiverScript)) { Write-Host "ERROR: Not found: $ReceiverScript"; exit 1 }
if (!(Test-Path -LiteralPath $PlotScript))     { Write-Host "ERROR: Not found: $PlotScript"; exit 1 }

$ReceiverWorkDir = Split-Path -Parent $ReceiverScript
$PlotWorkDir     = Split-Path -Parent $PlotScript

# ----------------
# Local cleanup prompt
# ----------------
if (Test-Path -LiteralPath $RadarNewDir) {
    $items = @(Get-ChildItem -LiteralPath $RadarNewDir -Force -ErrorAction SilentlyContinue)
    if ($items.Count -gt 0) {
        Write-Host "Found $($items.Count) item(s) in: $RadarNewDir"
        if (Ask-YesNo "Do you want to permanently delete them?") {
            Get-ChildItem -LiteralPath $RadarNewDir -Force -ErrorAction Stop |
                Remove-Item -Recurse -Force -ErrorAction Stop

            $left = @(Get-ChildItem -LiteralPath $RadarNewDir -Force -ErrorAction SilentlyContinue)
            Write-Host "Deleted contents of $RadarNewDir"
            Write-Host "Remaining items: $($left.Count)"
        } else {
            Write-Host "Kept existing files in $RadarNewDir"
        }
    } else {
        Write-Host "No files found in $RadarNewDir"
    }
} else {
    Write-Host "Warning: Folder not found: $RadarNewDir"
}

# ----------------
# Launch Receiver + Plotter (new PS windows)
# ----------------
$receiverCmd = "& { Set-Location -LiteralPath '$ReceiverWorkDir'; python '$ReceiverScript' }"
Start-Process powershell.exe -ArgumentList @("-NoExit","-Command",$receiverCmd)

$plotCmd = "& { Set-Location -LiteralPath '$PlotWorkDir'; python '$PlotScript' }"
Start-Process powershell.exe -ArgumentList @("-NoExit","-Command",$plotCmd)

# ----------------
# Remote cleanup prompt + run sender via WSL -> SSH
# ----------------
$doRemoteClean = Ask-YesNo "Remote: do you want to clean $RemoteFolder on $RemoteIP ?"

$remoteCmd = "cd $RemoteHome && " +
             ($(if ($doRemoteClean) { "rm -rf $RemoteFolder/* && " } else { "" })) +
             "python radar_sender.py"

wsl sh -lc "ssh -tt $SshOpts root@$RemoteIP '$remoteCmd'"

Write-Host ""
Write-Host "Finished."
