import subprocess
files = ["correlation.py", "Plots.py","statisticss.py"]
for script in files:
    subprocess.run(["python", script])