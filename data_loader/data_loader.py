import numpy as np
import torch
from scipy.stats import cauchy,multivariate_normal
from sklearn.model_selection import train_test_split


    
class generate_sample_MLP(torch.nn.Module):
    def __init__(self,d):
        super(generate_sample_MLP, self).__init__()
        torch.manual_seed(1)
        self.linear1 = torch.nn.Linear(d,1)
        #self.linear2 = torch.nn.Linear(10*d,1)

    def forward(self, x):
        layer1_out = torch.sin(self.linear1(x))
        #+self.linear1(x)
        #out = torch.sigmoid(self.linear2(layer1_out))
        out=(layer1_out+1)/2
        return out

class simulation_data_loader(object):
    def __init__(self,d,n_train,n_test):
        self.d=d
        self.n_train=n_train
        self.n_test=n_test
        np.random.seed(123)
    
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
    
    
def get_data():
    data=pd.read_csv("./data/Housing_californiya.csv", sep=',')
    data=data.dropna()  
    data=data[data["median_house_value"]<500000]
    
    dummy_data=pd.get_dummies(data.ocean_proximity, prefix='ocean_proximity')
    data.drop(["longitude","latitude","ocean_proximity"],axis=1,inplace=True)
    data_col_names=data.columns
    

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data))
    data.columns=data_col_names
    
 
    y_data=data["median_house_value"].values
    dummy_data.index=data.index
    x_data=pd.concat([data,dummy_data],axis=1).drop(["median_house_value"],axis=1).values
 

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=1)
    X_train=torch.tensor(X_train)
    X_test=torch.tensor(X_test)
    y_train=torch.tensor(y_train)
    y_test=torch.tensor(y_test)
    return X_train, X_test, y_train, y_test
