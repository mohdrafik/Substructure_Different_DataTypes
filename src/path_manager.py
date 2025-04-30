# src/path_manager.py
import sys
from pathlib import Path

def addpath():
    """
    use Case:
    1.Use It in Jupyter:
        from path_manager import addpath
        paths = addpath()
        # Use the returned dictionary if needed
        print("Base dir:", paths['BASE_DIR'])

    2. Use in Python Script:
        from path_manager import addpath
        addpath()
        # Then import modules

    Automatically adds 'src' and 'src/modules' to sys.path.
    Works in both Jupyter and .py scripts.
    Returns the base, src, and modules paths.
    """
    try:
        # Script mode
        base_dir = Path(__file__).resolve().parent.parent
    except NameError:
        # Jupyter mode — use notebook location
        base_dir = Path.cwd()
        # Adjust if running inside notebooks/ or src/
        if base_dir.name == 'notebooks' or base_dir.name == 'src':
            base_dir = base_dir.parent

    # Construct target paths
    src_path = base_dir / 'src'
    modules_path = src_path / 'modules'

    # Add to sys.path if not already there
    for path in [src_path, modules_path]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return {
        "BASE_DIR": base_dir,
        "SRC_PATH": src_path,
        "MODULES_PATH": modules_path
    }
