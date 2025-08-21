@echo off
cd /d "E:\project\gamma_hedge"
call E:\miniconda\condabin\conda activate learn
python scripts/run_tests.py %*