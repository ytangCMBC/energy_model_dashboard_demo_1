# Transit BEB Energy Dashboard (Demo)

This repository contains a **lightweight demo** of the Transit Battery Electric Bus (BEB) Energy Dashboard.  
It allows users to explore sample results through an interactive web app built with **Streamlit**. This repo is designed for demonstration and deployment on **Streamlit Community Cloud**.

---

## Features
- Interactive charts (energy, SOC, speed, grade)
- Route map with Folium
- Filter by Depot → Route → Direction → Duty → Shape ID
- Powered by precomputed data 

---

## Quick Start

### 1. Install environment

### Option 1 — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate transit-dashboard
```

### Option 2 — pip/venv
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run src/dashboard.py
```
- Open the link shown in the terminal (default: http://localhost:8501).
