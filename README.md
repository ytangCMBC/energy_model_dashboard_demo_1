# 🚍 Transit BEB Energy Dashboard  
### Route-Level & Block-Level Energy Analysis for Battery Electric Bus Simulations

This repository provides an **interactive multi-panel dashboard** to explore Battery Electric Bus (BEB) energy modeling results.  
The dashboard is structured around three analytical levels:

---

# 🧭 1. Route-Level Dashboard (Trip Analysis Panel)

This panel provides the **most basic** view of BEB performance along a single route (shape_id).  
It supports deep inspection of energy consumption patterns along the vehicle’s actual driven path.

### Features
- Matched GTFS route in an interactive **map**
- Movable **cursor** synchronized with:
  - Speed vs distance  
  - Elevation vs distance  
  - Grade vs distance  
  - SOC vs distance  
- Map overlays:
  - GTFS stops  
  - Traffic signals  
  - Stop signs  
  - Bridge spans based on elevation mask  
- KPIs:
  - Travel distance  
  - Energy consumption  
  - kWh/km  
  - Duty mode (medium/heavy)
- Supports **in-service** and **deadhead** trips

---

# 🧱 2. Block-Level Dashboard (Depot-Only Charging)

This panel aggregates individual trips into transit blocks and evaluates feasibility under **depot-only charging** (currently only for 40-ft bus).

### Features
- Block-level success/failure (SOC never below threshold)
- KPIs for heavy/medium duty modes  
- SOC remaining and energy use per block  
- Filters: depot → service day → line group → block number   
- Original `combined_sequence_json` shown in clean JSON format

Used to validate whether existing operations can run BEBs without on-route chargers.

---

# ⚡ 3. Block-Level Dashboard (On-Route Charging Simulation)

This panel evaluates block feasibility when **on-route chargers (ORC)** are available.

### Features
- Select/deselect ORC candidate locations  
- Automatic charging simulation based on SOC-dependent charge curve  
- Scenario caching for fast comparisons  
- KPIs:
  - Success rate before/after ORC  
  - Number of successful blocks  
  - Energy delivered  
- Depot-level ORC impact summaries  
- Per-block SOC trajectory plots  
- Same filtering workflow as depot-only panel

Ideal for **charging network design** and **scenario planning**.

---


