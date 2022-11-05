# EXPERIMENTS of NCF model in GBC dataset

## Performance of Generation Model
To reproduce the result in Figure 4:``python main_conditionVAE_control.py --user test``
To reproduce the result in Table 4:``python main_conditionVAE_faithful.py``

## Recommendation Refinement
To reproduce the result in Table 5: ``python main_conditionVAE_onetag.py``
Set test_pos in line 274 to True for "PosTag" and False for "NegTag" 


## Qualitative Evaluation of Explanation.
To get the distribution of Importance Scores in Figure 5: ``python main_conditionVAE_lime.py``


## Other
train_conditionVAE.py: train CVAE model