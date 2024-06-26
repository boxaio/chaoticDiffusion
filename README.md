# Chaotic Diffusion
The implementation code of the paper “Learning to Rectify the Probability Flow of Delay-induced Chaotic Diffusion with Action Matching”


# Generating the datasets

The configurations are listed in the file `exp_config.py`, including `chaoticDDE_n2_mix_gaussian_slow`,
`chaoticDDE_n2_mix_gaussian_fast`,
`chaoticDDE_n2_olympic_slow`,`chaoticDDE_n2_olympic_fast`

The datasets are generated by running the following command in the terminal:

```bash
python main.py
```

# Requirements

[JiTCDDE](https://github.com/neurophysik/jitcdde) is a just-in-time compilation for delay differential equations (DDEs). If allows one to integrate delay differential equations with adaptive step sizes and stiffness detection.

To install [JiTCDDE](https://github.com/neurophysik/jitcdde), run the following command in the terminal:
```bash
pip3 install jitcdde
```


# RNN experiment

The experiment files are in the folder `\RNN`
```bash
python runGRU.py
```
or
```bash
./runGRU.sh
```

# Neural ODE experiment

The experiment files are in the folder `\FP`
```bash
python main_fp_adj.py
```

# Action Matching experiment

```bash
python main_amcd.py
```

# References


[1] K. Neklyudov, R. Brekelmans, D. Severo, A. Makhzani, Action Matching: Learning Stochastic Dynamics from Samples,
International Conference on Machine Learning (2023) 25858–25889,arXiv:2210.06662.
[2] J. Losson, M. C. Mackey, R. Taylor, M. Tyran-Kami´nska, Density Evolution Under Delayed Dynamics: An Open Problem,
Vol. 38 of Fields Institute Monographs, Springer US, New York, 2020.
[3] P. Vlachas, J. Pathak, B. Hunt, T. Sapsis, M. Girvan, E. Ott, P. Koumoutsakos, Backpropagation algorithms and Reservoir
Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal dynamics, Neural Networks 126 (2020)
[4] L. Li, S. Hurault, J. Solomon, Self-Consistent Velocity Matching of Probability Flows, Advances in Neural Information Processing Systems 36,2023