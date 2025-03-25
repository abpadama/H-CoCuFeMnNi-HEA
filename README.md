# H-CoCuFeMnNi HEA systems

includes data and code on H-CoCuFeMnNi HEA systems of the paper 

### Machine learning and density functional theory-based analysis of the surface reactivity of high entropy alloys: The case of H atom adsorption on CoCuFeMnNi
Allan Abraham B. Padama, Marianne A. Palmero, Koji Shimizu, Tongjai Chookajorn, and Satoshi Watanabe

https://doi.org/10.1016/j.commatsci.2024.113480

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Abstract
This study examines the adsorption of H atom on CoCuFeMnNi(111) high entropy alloy (HEA) surface using a combination of density functional theory (DFT) and machine learning (ML) techniques. Hume-Rothery rule, thermodynamic parameters, and electronic structure analysis were utilized to elucidate the stability and reactivity of the CoCuFeMnNi surface. We found that CoCuFeMnNi is a stable solid solution with a fcc structure. By integrating surface microstructure-based input features into our ML model, we accurately predicted H adsorption energies on the hollow sites of CoCuFeMnNi surfaces. Our electronic properties analysis of CoCuFeMnNi revealed that there is an evident interaction among the elements, contributing to a broad range of adsorption energies. During adsorption, the nearest neighbor surface atoms to H directly engage with the adsorbate by transferring charge significantly. The atoms in other regions of the surface contribute through charge redistribution among the surface atoms, influencing overall charge transfer process during H adsorption. We also observed that the average of the d-band centers of the nearest neighbor surface atoms to H influence the adsorption energy, supporting the direct participation of these surface atoms toward adsorption. Our study contributes to a deeper understanding of the influence of surface microstructures on H adsorption on HEAs.


## Installation

You can download the data used in this study by cloning the git repository:
   ```sh
   git clone https://github.com/abpadama/H-CoCuFeMnNi-HEA.git
   ```

[//]: # (To install the required packages, use)

[//]: # (   ```sh)

[//]: # (   pip install -r requirement.txt)

[//]: # (   ```)

<!-- USAGE EXAMPLES -->
## Computational Details (DFT)
> **Software**: Quantum Espresso ver. 7.2
>
> **Pseudopotentials**: co_pbe_v1.2.uspp.F.UPF, cu_pbe_v1.2.uspp.F.UPF, fe_pbe_v1.5.uspp.F.UPF, mn_pbe_v1.5.uspp.F.UPF, ni_pbe_v1.4.uspp.F.UPF, h_pbe_v1.4.uspp.F.UPF
>
> **Kpoints**: 4 x 4 x 1 Monkhorst-Pack k-points mesh
>
> **Energy Cutoff**: 550 eV
>
> **Force Convergence Threshold**: 0.02 eV/Å
>
> **Slab Model**:  3x3x4 supercell of (111) facet, with 14 Å vacuum layer
>
> **Functional:**: BEEF-vdW
>
> **Adsorbate considered in this study**: H
>
> **Adsorption sites considered in this study**: fcc-hollow (fcc), and hcp-hollow (hcp)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Machine Learning
**Model**: Gaussian Process Regression
**Features**: count of atom per element per region (see data_h_[site]_element.csv)
**Target output**: adsorption energy ($E_{ads}$)
**Hyperparameter tuning**: Grid Search (see code_element.py)
**Prediction**: all possible combinations of microstructures (see combinations_h_[site].csv)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## Citation
Padama, A. A. B., Palmero, M. A., Shimizu, K., Chookajorn, T., & Watanabe, S. (2025). Machine learning and density functional theory-based analysis of the surface reactivity of high entropy alloys: The case of H atom adsorption on CoCuFeMnNi. Computational Materials Science, 247, 113480.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Email: abpadama@up.edu.ph

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This work is primarily funded by the Department of Science and Technology - Philippine Council for Industry, Energy and Emerging Technology Research and Development (DOST-PCIEERD) with Project No. 10128, 2022 (Project Title: Designing High Entropy Alloy Surfaces for Catalytic Applications using Atomistic Calculations and Materials Informatics Investigations (cHEApp)). This project is under the East Asia Science and Innovation Area Joint Research Program (e-ASIA JRP) with the title ‘‘Computational Design of High Entropy Alloys for Catalyst and BaTtery Applications (ACT)’’. This research was carried out using the High-Performance Computing Facility for Atomic Scale and Materials Informatics Investigation (HASMIN) of the Institute of Physics, University of the Philippines Los Baños, the Computing and Archiving Research Environment (COARE) of the Department of Science and Technology – Advanced Science and Technology Institute (DOST-ASTI), and the computational resources of the supercomputer Fugaku provided by the RIKEN Center for Computational Science.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
