import os
from pathlib import Path
import sys

class addPathModule:
        
    module_path = r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\src\modules"
    if not os.path.exists(module_path):
        print(f" i am looking for the Gaetano sys path")
        module_path = r"C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\modules"    
    # Add the module path to sys.path if it's nat already there
    print(f"module path: {module_path}")
    if module_path not in sys.path:
        sys.path.append(module_path)

