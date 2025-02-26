<# 
    AutoRestartClient.ps1
    Wrapper script for run_client.py that automatically restarts the client if it terminates.
    It keeps track of the number of restarts and aborts if there are too many within a certain time window.
    
    NOTE: To stop the script, you may need to close the PowerShell window.
    
    Configuration:
    - $THRESHOLD_TIME_WINDOW: Time window in seconds (e.g. 600 seconds = 10 minutes)
    - $THRESHOLD: Maximum number of restarts allowed within the time window.
    - $NTFY_TOPIC: If set, restart notifications will be sent to ntfy.sh/$NTFY_TOPIC.
#>



$env:COMPRL_SERVER_URL = "comprl.cs.uni-tuebingen.de"
$env:COMPRL_SERVER_PORT = "65335"
$env:COMPRL_ACCESS_TOKEN = "916cdb8c-bd47-4e2b-b86b-7d0c9e59b1c8"


# --- Activate the Python virtual environment ---
$venvActivatePath = "C:\UNI_Projekte\RL\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivatePath) {
    # Dot-source the activation script so that the environment is activated in the current session.
    . $venvActivatePath
    Write-Host "Activated Python virtual environment from $venvActivatePath"
} else {
    Write-Host "WARNING: Virtual environment activation script not found at $venvActivatePath"
}

# Configuration variables
$THRESHOLD_TIME_WINDOW = 600  # 10 minutes
$THRESHOLD = 10
$NTFY_TOPIC = ""  # Set your ntfy.sh topic if desired

# Array to hold termination timestamps (as Unix time)
$terminationTimes = @()

while ($true) {
    # Run the client (pass all provided arguments)
    python ./run_client.py @args

    # Get the current Unix timestamp (in seconds)
    $currentTime = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

    # Add the current timestamp to the array
    $terminationTimes += $currentTime

    # Filter out timestamps that are older than the defined time window
    $terminationTimes = $terminationTimes | Where-Object { $currentTime - $_ -le $THRESHOLD_TIME_WINDOW }

    # Check if the number of terminations exceeds the threshold
    if ($terminationTimes.Count -gt $THRESHOLD) {
        Write-Host ""
        Write-Host "##############################################"
        Write-Host "Restarted too many times within the last $THRESHOLD_TIME_WINDOW seconds."

        if ($NTFY_TOPIC -ne "") {
            Invoke-RestMethod -Uri "https://ntfy.sh/$NTFY_TOPIC" -Method Post `
                -Headers @{ "Priority" = "urgent"; "Tags" = "warning" } `
                -Body "Too many restarts within the last $THRESHOLD_TIME_WINDOW seconds"
        }

        exit 1
    }

    # Wait a bit before restarting (in case the client terminated due to a server restart)
    Start-Sleep -Seconds 20

    Write-Host ""
    Write-Host "##############################################"
    Write-Host "#                Restarting                  #"
    Write-Host "##############################################"
    Write-Host ""

    if ($NTFY_TOPIC -ne "") {
        Invoke-RestMethod -Uri "https://ntfy.sh/$NTFY_TOPIC" -Method Post -Body "Restarting"
    }
}
