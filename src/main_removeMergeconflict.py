# # from pathlib import Path

# # BASE_DIR = Path(__file__).resolve().parent  # Gets the folder containing this script
# # print("Base Directory:", BASE_DIR)

# import sys
# from pathlib import Path
# import os
# # import numpy as np

# # BASE_DIR = Path(__file__).resolve().parent.parent  # Moves 2 levels up to project root

# BASE_DIR1 = Path.cwd().parent.parent.parent/ "extendedReality_deepLearning" #  works for the jupyter notebook but not give exact file path in vs code as expected.


# print(f"base directory :---->  {BASE_DIR1} \n {os.listdir(BASE_DIR1)}")
# # MODULES_DIR = BASE_DIR / "src" / "modules"
# # print(f"MODULE DIRECORY : directory :{MODULES_DIR}")
# # sys.path.append(str(MODULES_DIR))
# # 
# # Now you can import from modules
# # from listspecificfiles import readlistFiles # type: ignore
# # 
# # files = readlistFiles(MODULES_DIR,keyword = 'mat')
# # print(files)




from pathlib import Path
import re
import os 
# Reload the notebook file since kernel was reset
filepath1= r"C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\garbage.ipynb"
filepath1 = os.path.normpath(filepath1)

notebook_path = Path(filepath1)

with open(notebook_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_content = f.read()

# Remove Git conflict markers and keep only the latest code blocks
conflict_pattern = re.compile(r"<<<<<<< HEAD.*?=======\n(.*?)>>>>>>>.*?\n", re.DOTALL)
cleaned_content = re.sub(conflict_pattern, r"\1", raw_content)

# Save the cleaned notebook
cleanedfile = r"C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\garbage_cleaned.ipynb"

cleanedfile = os.path.normpath(cleanedfile)
print(f"cleaned path file ------------> {cleanedfile}")
cleaned_path = Path(cleanedfile)

with open(cleaned_path, "w", encoding="utf-8") as f:
    f.write(cleaned_content)

cleaned_path.name
