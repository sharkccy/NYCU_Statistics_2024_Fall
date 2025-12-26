**Project Overview**
- This repository contains the Team 7 final project — a statistical analysis of regional data across different time periods. The analysis focuses on drunk driving accident indicators such as enforcement cases (takedowns), referrals to prosecution, accident counts, deaths, and injuries per 100,000 people. The code performs descriptive statistics, ANOVA, pairwise t-tests, and produces charts and maps where applicable. See `Group7_final_project.pdf` for a detailed report of the analysis and findings.

**Files**
- Program: [final.py](final_project/final.py)
- Presentation: [Group7_final_project_presentation.pdf](final_project/Group7_final_project_presentation.pdf)
- Report: [Group7_final_project.pdf](final_project/Group7_final_project.pdf)

**Environment & Dependencies**
- Python: Recommended Python 3.8 or later.
- Main libraries: `pandas`, `numpy`, `scipy`, `geopandas`, `imageio`, `matplotlib`, `geopandas`, `imageio`

```
pip install pandas numpy scipy geopandas imageio matplotlib
```

**Run Instructions**
1) Place the data files (`period_*.csv` and `period_all.csv`, etc.) in the `final_project` folder.
2) From the project folder run:

```bash
cd final_project
python final.py
```

3) The script computes means and variances for each indicator, performs ANOVA and pairwise t-tests, and prints results to the console. If `final.py` generates charts or maps, those will be displayed or saved depending on its implementation.

**Outputs**
- Console output of ANOVA statistics (F, p-value), pairwise test results (p-values and confidence intervals), and any generated plots or map images.

**Notes**
- Installing `geopandas` on Windows can be more involved due to binary dependencies. Using `conda` is recommended to simplify installation:

```
conda create -n stats_env python=3.10
conda activate stats_env
conda install -c conda-forge geopandas
pip install pandas scipy imageio matplotlib
```

