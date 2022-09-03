import numpy as np
import torch
from scipy.stats import cauchy,multivariate_normal
from sklearn.model_selection import train_test_split
from ..model import generate_sample_MLP


class simulation_data_loader(object):
    def __init__(self,d,n_train,n_test):
        self.d=d
        self.n_train=n_train
        self.n_test=n_test
    
    def get_train_data(self):
        X=multivariate_normal.rvs(size=self.d*self.n_train).reshape(-1,self.d)
        scale=np.sqrt((X**2).sum(axis=1))*cauchy.rvs(size=self.n_train)
        X=torch.tensor(X/scale.reshape(-1,1))

        generate_sample_model=generate_sample_MLP(self.d)
        Y=generate_sample_model(X).squeeze().detach()
        return X,Y
    
    def get_test_data(self):
        X=multivariate_normal.rvs(size=self.d*self.n_test).reshape(-1,self.d)
        scale=np.sqrt((X**2).sum(axis=1))*cauchy.rvs(size=self.n_test)
        X=torch.tensor(X/scale.reshape(-1,1))

        generate_sample_model=generate_sample_MLP(self.d)
        Y=generate_sample_model(X).squeeze().detach()
        return X,Y
