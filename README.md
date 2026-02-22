# Weather-driven US milk yield losses and economic damages revealed by 9 million cows

[![DOI](https://zenodo.org/badge/631829292.svg)](https://zenodo.org/doi/10.5281/zenodo.18735454)
Supporting code for Choi et al., (2026) 'Weather-driven US milk yield losses and economic damages revealed by 9 million cows'

Please contact Eunkyoung Choi at kyoung.choi@colostate.edu or ekchoi@bu.edu if you find any errors in the code or have questions.

## Repository Organization:
The project directory is organized into the following subdirectories:
- 1_data:
     - contains synthetic data to reproduce the analysis pipeline.
     - weather data used in the manuscript are publicly available from PRISM climate group and AgERA5.
     - milk records are from the Dairy Records Management Systems (DRMS) through a data sharing agreement. Due to a data confidentiality, these data cannot be made publicly available. Synthetic data mimicking the structure of original data are provided here with code.
    
- 2_code:
  - contains scripts for analyzing data and creating figures
 
- 2_utils:
  - contains code funtions
 
- 3_outputs:
  - save output and figures

- requirements.in: specific python packates needed to run codes
- requirements.txt: specific python package versions used for analysis
  
## Steps to setp up the environment and install python functions
### 1. Clone the repository
git clone https://github.com/kyoungyi/2025_Dairy_ML.git <br>
cd 2025_Dairy_ML

### 2. Create and activate a virtual environment
python3.10 -m venv .venv<br>
source .venv/bin/activate 

### 3. Install dependencies
pip install --upgrade pip<br>
pip install -r requirements.txt

### 4. Install the local utils package if a setup.py exists
pip install -e .

