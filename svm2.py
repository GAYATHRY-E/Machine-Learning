  #svm for a standard datasets load_iri from sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris 
iris= load_iris

x=iris.data
y=iris.target

#model

ml=SVC()

ml.fit(x,y)
result=ml.predict([5.1	,3.5,	1.4,	0.5])
result
