# EXPERIMENTS 4.1 Synthetic Data

## Performance of CVAE
To reproduce the result in Table 1:
CVAE: ``python main_conditionVAE.py --user test``
MLP: ``python main_mlp.py --user test``

## Performance of LIME
To reproduce the result in Table 2:
``python main_conditionVAE_lime.py --user test``

## Other
make_data2.py: generate synthetic data
train_conditionVAE.py: train CVAE model
train_mlp.py: train baseline MLP model