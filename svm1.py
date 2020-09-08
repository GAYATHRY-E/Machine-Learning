    #training 
    #data entering
import numpy as np
x=num.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y=num.array([1,1,2,2])
     #model creation
from sklearn.svm import svc
ml=svc()

ml.fit(x,y)

result=ml.predict([[1,-1]])
result
