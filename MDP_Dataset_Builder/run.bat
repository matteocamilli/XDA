@echo off

set MAX_SAMPLES=5000
set TOTAL_THREADS=16
set NUM_THREADS=0


python .\main.py --max-samples %MAX_SAMPLES%

:loop
if %NUM_THREADS% == %TOTAL_THREADS% goto end

START CMD /C "python .\main.py --index-to-run %NUM_THREADS% --total-executions %TOTAL_THREADS%" ^& PAUSE

set /a NUM_THREADS=%NUM_THREADS%+1
goto loop

:end
