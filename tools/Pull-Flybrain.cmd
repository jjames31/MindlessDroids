@echo off
setlocal
rem Run the locally reviewed helper, never an automatically replaced GitHub helper.
wsl.exe -d Ubuntu-22.04 -u jaco -- python3 -B /mnt/c/fba/.maintenance/github-sync/bin/flybrain_sync.py %*
exit /b %errorlevel%
