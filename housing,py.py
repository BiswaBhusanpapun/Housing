import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBooster:
    
    def __init__(self, max_depth=8, min_samples_split=5, min_samples_leaf=5, max_features=3, lr=0.1, num_iter=1000):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.lr = lr
        self.num_iter = num_iter
        self.y_mean = 0
        
    def __calculate_loss(self,y, y_pred):
        loss = (1/len(y)) * 0.5 * np.sum(np.square(y-y_pred))
        return loss
    
    def __take_gradient(self, y, y_pred):
        grad = (y-y_pred)
        return grad
    
    def __create_base_model(self, X, y):
        base = DecisionTreeRegressor(criterion='squared_error',max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_features=self.max_features)
        base.fit(X,y)
        return base
    
    def predictA(self,models,X):
        pred_0 = self.y_mean
       # pred = pred_0.reshape(len(pred_0),1)
        pred = pred_0
        for i in range(len(models)):
            temp = (models[i].predict(X))
            #print("temp :",temp.shape)
            pred += self.lr * temp
        
        return pred
    
    
    def predict(self,models,y,X):
        pred_0 = np.array([self.y_mean] * len(X))
        pred = pred_0.reshape(len(pred_0),1)
        
        for i in range(len(models)):
            temp = (models[i].predict(X)).reshape(len(X),1)
            pred += self.lr * temp
        
        return pred
    
   
    
    def train(self, X, y):
        models = []
        losses = []
        self.y_mean = np.mean(y)
        #print("np.Mean :", np.mean(y))
        pred_0 = np.array([np.mean(y)] * len(y))
        #print("pred_0:", pred_0)
        pred = pred_0.reshape(len(pred_0),1)
        #print("pred:", pred)
        
        for epoch in range(self.num_iter):
            loss = self.__calculate_loss(y, pred)
            losses.append(loss)
            grads = self.__take_gradient(y, pred)
            base = self.__create_base_model(X, grads)
            r = (base.predict(X)).reshape(len(X),1)
            #pred = pred - alpha*r
            pred += self.lr * r
            models.append(base)
            
        return models, losses, pred_0
    
    
    
    
    
    
    
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as pt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#READ DATA
data = pd.read_csv("housing.csv")
data.fillna(0,inplace=True)
#X,y
X = data.iloc[:,1:5]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)
#scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = np.array(y_train).reshape(X_train.shape[0],1)
y_test = np.array(y_test).reshape(X_test.shape[0],1)
#TRAIN
G = GradientBooster()
models, losses, pred_0 = G.train(X_train,y_train)



x_input = [[6,148,72,35,0,33.6,0.627,50]]
x_input = data.iloc[:,1:5]
X_transformed = scaler.fit_transform(x_input)
dfX = pd.DataFrame(X_transformed)
predictions_output = G.predictA(models, X_train[0:1])
print("predictions_output",predictions_output)
print("actual_____output",y_train[0:1])


sns.set_style('darkgrid')
ax = sns.lineplot(x=range(1000),y=losses)
ax.set(xlabel='Epoch',ylabel='Loss',title='Loss vs Epoch')




#print("x_train:",y_test.shape)
#y_pred = G.predict(models, y_test, X_test)
#print('Prediction :',y_pred)
#print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))
#RMSE: 49396.079511786884






















