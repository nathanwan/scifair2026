@echo off
title Ballistics Simulator Runner
echo Checking dependencies...

:: Check if Streamlit is installed, if not, install it
python -m pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit not found. Installing...
    pip install streamlit numpy matplotlib
)

echo Starting the simulator...
:: Run the streamlit app
streamlit run demo_v1.0.2.py

pause