# Chaotic Diffusion
The implementation code of the paper “Learning to Rectify the Probability Flow of Delay-induced Chaotic Diffusion with Action Matching”


Generating the datasets
------------------
```bash
python3 main.py
```

Requirements
-------------------
[JiTCDDE](https://github.com/neurophysik/jitcdde) is an just-in-time compilation for delay differential equations (DDEs). If allows one to integrate delay differential equations with adaptive step sizes and stiffness detection.

To install JiTCDDE, run the following command in the terminal:
```bash
pip3 install jitcdde
```


References
-------------------

[1] K. Neklyudov, R. Brekelmans, D. Severo, A. Makhzani, Action Matching: Learning Stochastic Dynamics from Samples,
International Conference on Machine Learning (2023) 25858–25889,arXiv:2210.06662.