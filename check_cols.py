import re, sys
with open("venv/Lib/site-packages/project_x_py/realtime_data_manager/core.py", encoding="utf-8") as f:
    lines = f.readlines()
# find lines mentioning column names / schema
for i, ln in enumerate(lines):
    if any(x in ln for x in ['"open"', '"close"', '"high"', '"low"', '"timestamp"', '"time"', 'schema', 'columns', 'pl.col', 'bar_data']):
        print(f"{i+1}: {ln.rstrip()[:120]}")
