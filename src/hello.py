# src/hello.py
import sys
import platform
import pandas as pd

print("Hello — project environment is ready!")
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Pandas version:", pd.__version__)

git --version