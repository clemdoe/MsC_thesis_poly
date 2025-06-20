# Thesis Project: Development and Validation of a Simplified Thermal Hydraulics solver for the modelling of Boiling Water Reactor fuel channels in DONJON5  
**Author**: Clément Huet
**Supervisor**: Alain Hébert, Cédric Beguin
**Institution**: Polytechnique Montréal

---

## 📌 Overview  
This repository contains all data, code, and postprocessing scripts related to my thesis:  
- **GeN-Foam** simulations (including input and output).  
- **Open Source thermohydraulic code** THMprototype to model the thermohydraulics of BWR created during my master thesis
- **Postprocessing** scripts generating figures in the thesis.
- **Comparison data** between GeN-Foam and the prototype

---

## 🗂 Folder Structure  
- `0_THMprototype/`: developed to model the themrohydraulics of BWR.  
- `1_GeNFoam/`: simulations input and output files.  
- `2_Output/`: scripts generating figures in the thesis and output of the GeN-Foam to compare GeN-Foam and THMprototype. It is the input for the THM python prototype and for some case containing the output of GeN-Foam simulations to do the comparison.
- `3_Version5/`: the updated Version5 code to port the code THM_prototype to the THM: module of Donjon5
- `4_Figures/Results`: figures showed in my thesis.
- `5_docs/`: Thesis PDF and supporting references (publications, documentation).
---

## 🛠 How to Reproduce Results  
### Thermo-hydraulic Python Simulations
1. Install the dependancies mentionned in requirement.txt
2. Copy the test case file from `2_Output/` to `0_THMprototype/`
2.bis. To compare with results from other codes. Check the paths for the GeNFoam/TwoPorFlow/BFBT comparison. For example to compare with GeN-Foam results the relative path should be: `your_absolute_path/1_GeNFoam/test_case_studied` and you can use the GF_Plotter python class.
4. Change the abosolute paths to save your figures from mine to yours.
5. Run cases from `0_THMprototype/` with your python installation. You may need jupyter notebook. Without jupyter notebook you will need to copy paste the code in a .py file.

#### Run new GeN_Foam simulations
1. Install and compile GeN-Foam (see [official documentation](https://gitlab.com/foam-for-nuclear/GeN-Foam)).
2. Copy the studied case from 1_GeNFoam/` to your local GeN-Foam installation in the folder /run/
3. Your can modified the properties inside the different sub folders
4. Run with ./Allrun

---

## ✨ Multiphysics
For the multiphysics simulations see [this repository](https://github.com/clemdoe/BWR-multiphysics) and ask for permitions.
