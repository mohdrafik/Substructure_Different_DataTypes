from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent 
SCRIPT_DIR1 = Path(__file__).resolve() 

Python_runDIR = Path.cwd().parent
Python_runDIR1 = Path.cwd()

print(f"\n Script path is: {SCRIPT_DIR} \n and Script path without parent : {SCRIPT_DIR1}")

print(f"\n Same Script but different Command cwd path is: {Python_runDIR} \n and Same Script path without parent : {Python_runDIR1} \n")

