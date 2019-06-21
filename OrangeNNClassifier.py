# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:46:20 2018

@author: prysie
"""
import pandas as pd
import datetime
from Orange.data import Table, Domain
from Orange.classification import NNClassificationLearner
from Orange.evaluation import CrossValidation, scoring
from Orange.preprocess import Normalize, Scale
from Orange.preprocess import DomainDiscretizer
from Orange.preprocess.discretize import EqualWidth
 
raw_data_table = Table.from_file("white wine.csv")

feature_vars = list(raw_data_table.domain.variables[1:11])
#Bucket the classifier into distinct bins
discretizer = DomainDiscretizer()
discretizer.method = EqualWidth(n=9)
discretizer_domain = discretizer(raw_data_table)
class_label_var = discretizer_domain[0]
print(class_label_var.values)
wine_domain = Domain(feature_vars, class_label_var)
data_table = Table.from_table(domain=wine_domain, source=raw_data_table)
Table.save(data_table, "data_table normal2.csv")
def normalize_table(table_to_process):
    norm = Normalize(norm_type=Normalize.NormalizeBySpan)
    norm.transform_class = False
    norm_data_table = norm(table_to_process)
    norm_data_table.shuffle()
    return norm_data_table
#Normalise the feature values
norm_data_table = normalize_table(data_table)
print("Applying learner on total data records {}".format(len(norm_data_table)))

#Create a NN classifier learner
then = datetime.datetime.now()
ann_learner = NNClassificationLearner(hidden_layer_sizes=(10, ),max_iter=4000 )
ann_classifier = ann_learner(norm_data_table) 

#Do the 10 folds cross validation 
eval_results = CrossValidation(norm_data_table, [ann_learner], k=10)
now = datetime.datetime.now()
tdelta = now - then
print("Processing completed after: {} ".format(tdelta))

#Accuracy and area under (receiver operating characteristic, ROC) curve (AUC)
print("Accuracy: {:.3f}".format(scoring.CA(eval_results)[0]))
print("AUC: {:.3f}".format(scoring.AUC(eval_results)[0]))

#Remove minority classes
value_counts = pd.Series(raw_data_table[:,0]).value_counts()[pd.Series(raw_data_table[:,0]).value_counts() > 10]
value_counts1 = pd.Series(raw_data_table[:,0]).value_counts()[pd.Series(raw_data_table[:,0]).value_counts() < 10]
print(value_counts1)
first_elts = [x[0] for x in value_counts.keys().tolist()]
sel = [i for i, d in enumerate(raw_data_table) if d["quality"] in first_elts]
subset_data = raw_data_table[sel]
subset_data_table = Table.from_table(domain=wine_domain, source=subset_data)
norm_subset_data_table = normalize_table(subset_data_table)
print("Applying learner on filtered data recordset, total data records {}".format(len(norm_subset_data_table)))

then = datetime.datetime.now()
ann_learner = NNClassificationLearner(hidden_layer_sizes=(10, ),max_iter=4000 )
ann_classifier = ann_learner(norm_subset_data_table)  
eval_results = CrossValidation(norm_subset_data_table, [ann_learner], k=10)
now = datetime.datetime.now()
tdelta = now - then

print("Processing completed after: {} ".format(tdelta))
print("Accuracy: {:.3f}".format(scoring.CA(eval_results)[0]))
print("AUC: {:.3f}".format(scoring.AUC(eval_results)[0]))
