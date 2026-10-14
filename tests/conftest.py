import os
import sys

# Allow `import src...` when pytest is run from anywhere (the repo is not pip-installed by default).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
