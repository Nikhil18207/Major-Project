# Keep Windows awake during training
# Run this in a separate PowerShell window

$keepAwake = $true
Write-Host "🔥 Keeping system awake during training..."
Write-Host "Press Ctrl+C to stop"

while ($keepAwake) {
    # Send a dummy keystroke to prevent sleep
    $wsh = New-Object -ComObject WScript.Shell
    $wsh.SendKeys('+{F15}')  # Send Shift+F15 (doesn't do anything visible)

    # Wait 60 seconds
    Start-Sleep -Seconds 60
}
