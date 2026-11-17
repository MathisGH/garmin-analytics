<#
register_pipeline_task.ps1 -- register the daily Garmin pipeline in Windows Task Scheduler.

Goal: run run_pipeline.py every day at 21:00.

How: builds a scheduled task from four parts (what to run, when, with which
settings, as which user) and registers it.
#>

$ProjectRoot = Split-Path -Parent $PSScriptRoot   # this file lives in utils/
$TaskName    = "GarminDailyPipeline"
$PythonExe   = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PipelineScript = Join-Path $ProjectRoot "run_pipeline.py"

# 1. WHAT to run: the venv Python on the pipeline script, started in the project folder
$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$PipelineScript`"" `
    -WorkingDirectory $ProjectRoot

# 2. WHEN: every day at 21:00
$trigger = New-ScheduledTaskTrigger -Daily -At "21:00"

# 3. SETTINGS: catch up if missed, run on battery, give up after 1 hour
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# 4. WHO: my own account, only while I am logged on (no password stored)
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Daily update of the Garmin project (extract, anomaly scores, sleep analysis)" `
    -Force