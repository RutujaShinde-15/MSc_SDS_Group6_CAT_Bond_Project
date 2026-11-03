# Runs the whole China replication in order (about 10 minutes in total).
import subprocess, sys
for s in ["00_compile.py", "02_descriptive.py", "03_thresholds.py", "04_marginals.py",
          "05_copula.py", "06_frequency.py", "07_price.py", "08_section3.py"]:
    print(f"\n######## {s} ########", flush=True)
    subprocess.run([sys.executable, s], check=True)
