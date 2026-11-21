@echo off
rem Main evolution run (resumes from runs\main\state.pkl). Started standalone by sim\start_background.ps1.
cd /d "%~dp0.."
".venv\Scripts\python.exe" sim\evolve.py --out runs\main --gens 2000 --sigma 0.1 --ffa 8 --prac 2 --bench 25 --workers 24 --resume >> runs\main.log 2>&1
