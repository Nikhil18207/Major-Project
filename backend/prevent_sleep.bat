@echo off
echo Preventing Windows from sleeping during training...
echo.

REM Disable sleep and hibernate for AC power
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0

echo [OK] Power settings configured:
echo   - Standby: DISABLED
echo   - Hibernate: DISABLED
echo   - Monitor timeout: DISABLED
echo.
echo Your PC will stay awake during training.
echo Training expected to finish around 1:00 PM tomorrow.
echo.
pause
