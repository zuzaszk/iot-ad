import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

with open('all_cv_results.json', 'r') as f:
    cv_results = json.load(f)

with open('best_params.json', 'r') as f:
    best_params = json.load(f)

with open('final_test_evaluation.json', 'r') as f:
    test_results = json.load(f)
    
cv_data = []
for result in cv_results:
    params = result['params']
    params['avg_false_positive_rate'] = result['avg_false_positive_rate']
    cv_data.append(params)

df = pd.DataFrame(cv_data)


