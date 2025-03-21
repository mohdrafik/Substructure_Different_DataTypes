# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent  # Gets the folder containing this script
# print("Base Directory:", BASE_DIR)

import sys
from pathlib import Path
import os
# import numpy as np

# BASE_DIR = Path(__file__).resolve().parent.parent  # Moves 2 levels up to project root

BASE_DIR1 = Path.cwd().parent.parent.parent/ "extendedReality_deepLearning" #  works for the jupyter notebook but not give exact file path in vs code as expected.


print(f"base directory :---->  {BASE_DIR1} \n {os.listdir(BASE_DIR1)}")
# MODULES_DIR = BASE_DIR / "src" / "modules"
# print(f"MODULE DIRECORY : directory :{MODULES_DIR}")
# sys.path.append(str(MODULES_DIR))
# 
# Now you can import from modules
# from listspecificfiles import readlistFiles # type: ignore
# 
# files = readlistFiles(MODULES_DIR,keyword = 'mat')
# print(files)




