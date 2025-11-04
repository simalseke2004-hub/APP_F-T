# 🧪 SpectrumFit — Minimal Setup Guide

This README explains how to **install only the required imports** to run `SpectrumFit_v13_SaveFix_FIX6.py` in any terminal —  
on **Windows**, **macOS**, or **Linux** — without installing unnecessary packages.

The setup uses a **virtual environment (venv)** to keep dependencies clean and isolated.  
Once installed, you can run the program directly with a single command.

---

## ⚡ Quick Summary (Windows)

1. Open **PowerShell** in the project folder  
2. Run this full command:
   ```powershell
   python -m venv .venv; . .\.venv\Scripts\Activate.ps1; pip install --upgrade pip wheel setuptools; pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 matplotlib==3.7.3 tkinterdnd2==0.3.0
   ```
3. Then start the GUI:
   ```powershell
   python SpectrumFit_v13_SaveFix_FIX6.py
   ```

✅ That’s it — SpectrumFit will open normally.

---

## 📦 Required Libraries

### 🧩 Third-party dependencies (need to be installed)
| Library | Version | Purpose |
|----------|----------|----------|
| `numpy` | 1.26.4 | Numerical calculations |
| `pandas` | 2.0.3 | Data handling / CSV I/O |
| `scipy` | 1.10.1 | Curve fitting and optimization |
| `matplotlib` | 3.7.3 | Plotting |
| `tkinterdnd2` | 0.3.0 | Drag & drop support for Tkinter GUI |

### 🏗️ Built-in Python modules (no installation needed)
`tkinter`, `os`, `sys`, `math`, `re`, `datetime`, `platform`,  
`traceback`, `concurrent.futures`, `ctypes`, `dataclasses`, `typing`, `types`

---

## 🪟 Windows Setup (PowerShell)

> 💬 **Commented version — safe to paste line-by-line**

```powershell
# 1️⃣ Create a local environment named ".venv"
python -m venv .venv

# 2️⃣ Activate it (must be done before installing)
. .\.venv\Scripts\Activate.ps1

# 3️⃣ Upgrade base tools
pip install --upgrade pip wheel setuptools

# 4️⃣ Install the required imports for SpectrumFit
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 matplotlib==3.7.3 tkinterdnd2==0.3.0

# 5️⃣ Run the program
python SpectrumFit_v13_SaveFix_FIX6.py
```

> 💡 If PowerShell blocks activation, run this once as admin:
> ```powershell
> Set-ExecutionPolicy RemoteSigned
> ```

---

## 🍎 macOS Setup (bash or zsh)

> 💬 **Commented version — paste line-by-line into Terminal**

```bash
# 1️⃣ Create a virtual environment
python3 -m venv .venv

# 2️⃣ Activate it
source .venv/bin/activate

# 3️⃣ Upgrade pip and base tools
pip install --upgrade pip wheel setuptools

# 4️⃣ Install dependencies
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 matplotlib==3.7.3 tkinterdnd2==0.3.0

# 5️⃣ Run the app
python3 SpectrumFit_v13_SaveFix_FIX6.py
```

> 🧠 If you get an error like `ModuleNotFoundError: No module named 'tkinter'`, install Tk:
> ```bash
> brew install tcl-tk
> ```

---

## 🐧 Linux Setup (bash)

> 💬 **Commented version — paste line-by-line into Terminal**

```bash
# 1️⃣ Install tkinter (if not already available)
sudo apt-get update && sudo apt-get install -y python3-tk

# 2️⃣ Create a virtual environment
python3 -m venv .venv

# 3️⃣ Activate it
source .venv/bin/activate

# 4️⃣ Upgrade core tools
pip install --upgrade pip wheel setuptools

# 5️⃣ Install dependencies
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 matplotlib==3.7.3 tkinterdnd2==0.3.0

# 6️⃣ Run the script
python3 SpectrumFit_v13_SaveFix_FIX6.py
```

---

## 🔍 Optional — Verify that all imports are working

Run this quick diagnostic inside your environment:
```bash
python - << 'PY'
import sys
mods = ["numpy","pandas","scipy","matplotlib","tkinterdnd2","tkinter"]
for m in mods:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}", file=sys.stderr)
PY
```

If all lines show `[OK]`, your setup is perfect.

---

## 🧹 Deactivate and Clean Up

Deactivate environment (when done):
```bash
deactivate
```

Remove environment completely:
```bash
# macOS / Linux
rm -rf .venv

# Windows (PowerShell)
rmdir /s /q .venv
```

---

## ✅ Final Note

After setup, simply run:

```bash
python SpectrumFit_v13_SaveFix_FIX6.py   # Windows
python3 SpectrumFit_v13_SaveFix_FIX6.py  # macOS / Linux
```

Your SpectrumFit GUI will launch with all required imports installed —  
ready for curve fitting, Gaussian analysis, and spectrum visualization.

---

**Author:** Şimal Şeker  
**Project:** Monochromator SpectrumFit GUI (v13 SaveFix)  
**Environment:** Python 3.9+ (tested on Windows 7/10, macOS Ventura, Ubuntu 22.04)
