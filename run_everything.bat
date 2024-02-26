@echo off

:: Check if two arguments are given
if "%~3"=="" (
    echo Usage: %0 ^<filename^> ^<config_id^>
    exit /b 1
)

:: Assign arguments to variables
set "FILE=%1"
set "CONFIG_ID=%2"

:: Run the segmentation script
python segment/main.py --file "%FILE%" --config_id "%CONFIG_ID%"

:: Run the tracking script
python track/track.py --file "%FILE%" --config_id "%CONFIG_ID%"

:: Run the postprocessing script
python track/postprocess.py --file "%FILE%" --config_id "%CONFIG_ID%"

:: Display the results
python display.py --file "%FILE%" --config_id "%CONFIG_ID%"