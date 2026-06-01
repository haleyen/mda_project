**MDA Project Report — Group 9**  
Modern Data Analytics (G0Z39a) · Academic Year 2025–2026


## Team
 
Tuan Anh Trinh · Van Ha Pham · Harper Le · Thuy Linh Hoang

# Bicycle Traffic Volume Estimation in Flanders
 
 
## Overview
 
A machine learning framework to estimate hourly bicycle traffic volumes across Flanders using a sparse network of automated counting stations. The model combines sensor data with weather, population density, infrastructure, and POI features to generate street-level traffic estimates for unmonitored roads.
 
## Results
 
The final LightGBM model (43 features) achieves:
- **Test MAE: 13.297** cyclists/hour (RMSE 35.969)
- **Validation MAE: 6.367** cyclists/hour (RMSE 17.718)
## Repository Structure
 
```
├── final_submission_group9/   # Final model code and notebooks
├── explo/                     # Exploratory analysis and dashboard
└── requirements.txt           # Python dependencies
```
 
## Setup
 
```bash
pip install -r requirements.txt
```
 
Developed with Python 3.12.11. Key dependencies: `lightgbm==4.6.0`, `scikit-learn==1.6.1`, `pandas==2.2.3`.
 
## Live Dashboard
 
Interactive traffic map for Leuven:  
[haleyen.github.io/mda_project/explo/andy/output/leuven_traffic_map_standalone.html](https://haleyen.github.io/mda_project/explo/andy/output/leuven_traffic_map_standalone.html)
 
