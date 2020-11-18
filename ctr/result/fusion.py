import pandas as pd
import numpy as np

res1 = pd.read_csv('submission_4_all.csv')
res2 = pd.read_csv('submission_4fold_win.csv')
res3 = pd.read_csv('submission_4_all_nounique.csv')

res1['probability'] = res1['probability'] * 0.6 + res2['probability'] * 0.2 + res3['probability'] * 0.2
res1['probability'] = res1['probability'].astype(np.float32)
res1.to_csv('submission_f.csv',index = False)