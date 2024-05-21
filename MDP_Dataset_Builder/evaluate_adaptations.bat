@echo off

set TOTAL_THREADS=1
set /a LAST_THREAD=%TOTAL_THREADS%-1
set NUM_THREADS=0
set PATH_TO_DATASET=%1

:loop
if %NUM_THREADS% == %LAST_THREAD% goto end

START CMD /C "python .\main.py --index-to-run %NUM_THREADS% --total-executions %TOTAL_THREADS% --path-to-dataset %PATH_TO_DATASET%"

set /a NUM_THREADS=%NUM_THREADS%+1
goto loop

:end
START /WAIT CMD /C "python .\main.py --index-to-run %NUM_THREADS% --total-executions %TOTAL_THREADS% --path-to-dataset %PATH_TO_DATASET%"

CMD /C "python merge_csvs.py"
