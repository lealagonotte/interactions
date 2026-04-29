# Interactions

This repository contains our final project for the Interactions course, taught by Julien Randon-Furling at the MVA, ENS Paris-Saclay. It focuses on wildfire propagation using cellular automata.

## Project Goal

This project studies wildfire propagation at two complementary scales.

At the **macroscopic scale**, we analyze the temporal succession of fires using the Drossel-Schwabl cellular automaton, in order to capture the large-scale dynamics that emerge over time.

At the **microscopic scale**, we focus on the propagation of a single fire event. We study how environmental factors such as wind, topography, vegetation age, and moisture influence the shape and speed of the fire front. The project also investigates how these models can be calibrated using synthetic and real wildfire data in order to improve their predictive capabilities.

## Main Contributions

- Reimplementation of cellular automata models for wildfire propagation.
- Numerical experiments to reproduce and interpret propagation dynamics.
- Extension of the basic models with environmental parameters such as vegetation age and moisture.
- Calibration experiments on synthetic and real data for fire-spread prediction.

## Code architecture
```text
interactions-main/
├── cell_auto_documents/          
├── data/
│   └── backtest/
│       └── hist_data.parquet     
├── notebooks/                    
│   ├── cellular_automaton.ipynb
│   ├── CA_Kara_age_peterson.ipynb
│   ├── CA_kara_humidity.ipynb
│   ├── drossel_schwabl_sim.ipynb
│   ├── synthetic_data.ipynb
│   ├── gradient_solver.ipynb
│   ├── gradient_solver_real_data.ipynb
│   ├── grid_search.ipynb
│   ├── grid_search _real_data.ipynb
│   └── data_enhancement.ipynb
├── simulators/                   
│   ├── CellularAutomaton.py
│   ├── CA_modified.py
│   ├── drossel_schwabl_CA.py
│   ├── model_solver.py
│   ├── model_solver_real_data.py
│   ├── metrics.py
│   └── backtest.py
├── pixi.toml                    
├── requirements.txt             
└── README.md
```

## Simulators

Contains the python files that code several simulators that are used in the example.

## Notebooks

This folder contains example notebooks that uses the classes from the simulator folder. 

## References

## References

1. Alexandridis, A., Russo, L., Vakalis, D., Bafas, G. V., and Siettos, C. I.  
   Wildland fire spread modelling using cellular automata: evolution in large-scale spatially heterogeneous environments under fire suppression tactics.  
   *International Journal of Wildland Fire*, 20, 633–647, 2011.

2. Drossel, B., and Schwabl, F.  
   Self-organized critical forest-fire model.  
   *Physical Review Letters*, 69, 1629–1632, 1992.

3. Grassberger, P.  
   Critical behaviour of the Drossel-Schwabl forest fire model.  
   *New Journal of Physics*, 4, 17–17, 2002.

4. Karafyllidis, I., and Thanailakis, A.  
   A model for predicting forest fire spreading using cellular automata.  
   *Ecological Modelling*, 99(1), 87–97, 1997.

5. Karakonstantis, I., and Xylomenos, G.  
   A review of two-dimensional cellular automata models for wildfire simulation: methods, capabilities, and limitations.  
   *Fire*, 9(3), 2026.

6. Mandel, J., Amram, S., Beezley, J. D., Kelman, G., Kochanski, A. K., Kondratenko, V. Y., Lynn, B. H., Regev, B., and Vejmelka, M.  
   Recent advances and applications of WRF-SFIRE.  
   *Natural Hazards and Earth System Sciences*, 14(10), 2829–2845, 2014.

7. Rothermel, R. C.  
   A mathematical model for predicting fire spread in wildland fuels.  
   Technical report, USDA Forest Service, Intermountain Forest and Range Experiment Station, Ogden, UT, 1972.

8. Von Neumann, J.  
   The general and logical theory of automata.  
   In L. A. Jeffress, editor, *Cerebral Mechanisms in Behavior: The Hixon Symposium*.  
   Wiley, New York, 1951.

9. White, S. H., Martín del Rey, A., and Rodríguez Sánchez, G.  
   A cellular automata model for predicting fire spread.  
   *International Journal of Wildland Fire*, 18(1), 1–12, 2009.

10. Zinck, R. D., and Grimm, V.  
    Unifying wildfire models from ecology and statistical physics.  
    *The American Naturalist*, 174(5), E170–E185, 2009.*


## Authors
- Gilles Févry - gilles.fevry@ensae.fr
- Léa Lagonotte - lea.lagonotte@ensae.fr
- Perann Nedjar - perann.nedjar@ensae.fr

