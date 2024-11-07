import torch
import pickle
import numpy as np
import pandas as pd
import sklearn as sk
import concurrent.futures
from sklearn import impute, preprocessing, metrics, model_selection
from sklearn.experimental import enable_halving_search_cv
import matplotlib.pyplot as plt

class Regressor():

    def __init__(self, x, nb_epoch = 2300, neural_architecture= [8,8,8],learning_rate=0.001, batch_size=50):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.x = x
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.x_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        self.string_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.x_normalizer = preprocessing.MinMaxScaler()
        self.y_normalizer = preprocessing.MinMaxScaler()
        self.label_binarizer = preprocessing.LabelBinarizer()
        self.batch_size=batch_size
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.nb_epoch = nb_epoch 
        self.output_size = 1
        self.learning_rate=learning_rate
        self.neural_architecture=neural_architecture
        layers = []
        input_size = self.input_size

        for hidden_layer in neural_architecture:
            layers.append(torch.nn.Linear(input_size, hidden_layer)) # Use Linear activation functions only
            input_size = hidden_layer
            
        layers.append(torch.nn.Linear(input_size, self.output_size))#final activation function                  
        self.model = torch.nn.Sequential(*layers)
        self.loss = torch.nn.MSELoss()#MSE loss

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        if training: self.x = x
        x_no_strings = x.drop(columns = ['ocean_proximity'])
        x_strings = x['ocean_proximity'].to_frame()
        columns=['households','population','total_bedrooms','total_rooms']
        x_no_strings[columns] = x_no_strings[columns].apply(np.log1p)#redistribute long tails
        if training:
            x_no_strings = self.x_imputer.fit_transform(x_no_strings)
            x_strings = self.string_imputer.fit_transform(x_strings) 
            x_strings=self.label_binarizer.fit_transform(x_strings)
            x_no_strings=self.x_normalizer.fit_transform(x_no_strings)
            if y is not None:
                y=self.y_normalizer.fit_transform(y)
        else:
            x_no_strings = self.x_imputer.transform(x_no_strings)
            x_strings = self.string_imputer.transform(x_strings) 
            x_strings=self.label_binarizer.transform(x_strings)
            x_no_strings=self.x_normalizer.transform(x_no_strings)
            if y is not None:
                y=self.y_normalizer.transform(y)

        x=np.concatenate((x_no_strings,x_strings),axis=1)
        x=torch.tensor(x)
        y=torch.tensor(y) if y is not None else None       
        return x , y
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y, training = True) # Do not forget
        n_x=X.size()[0]
        optimiser = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        for epoch in range(self.nb_epoch):
            optimiser.zero_grad()
            y_pred = self.model(X)
            loss = self.loss(y_pred, Y)
            loss.backward()
            optimiser.step()
        return self
    
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        model_output=self.model(X)
        denormalized_output=self.y_normalizer.inverse_transform(model_output.detach().numpy())
        return denormalized_output


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        y_real=y
        y_pred = self.model(X)
        y_pred_denormalized=self.y_normalizer.inverse_transform(y_pred.detach().numpy())
        RMSE=metrics.mean_squared_error(y_real,y_pred_denormalized,squared=False)
        return RMSE

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def RegressorHyperParameterSearch(x_dev, y_dev, x_train, y_train):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a multithreaded grid search hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Validation data sets and Training dataset 
        
    Returns:
        The function returns the optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    learning_rates = []
    counter = 0.0001 #learning rate increment
    starter = 5
    for i in range(1, 10):
        learning_rates.append(starter * counter)
        starter += 1

    neural_architectures = [[8,8,8], [8,8,6],[8,6,8]] #neural architecutres 

    best_score = float('inf')
    best_params = {}

    def evaluate_params(params):
        learning_rate, neural_architecture, nb_epoch = params
        regressor = Regressor(x_train, nb_epoch, neural_architecture, learning_rate)
        regressor.fit(x_train, y_train)
        train_score=regressor.score(x_train,y_train)
        dev_score = regressor.score(x_dev, y_dev)

        nonlocal best_score, best_params
        if dev_score < best_score:
            best_score = dev_score
            best_params = {
                'learning_rate': learning_rate,
                'neural_architecture': neural_architecture,
                'nb_epoch': nb_epoch
            }
        with open('finaltest.csv', 'a') as f:
            f.write(str(train_score) + ',' + str(dev_score) + ','+ str(learning_rate) + ',' + str(neural_architecture) + ',' + str(nb_epoch) + '\n')
        
    # Create a list of parameter combinations to evaluate
    parameter_combinations = [(lr, na, ne) for lr in learning_rates for na in neural_architectures for ne in range(500, 5000, 100)] #epochs

    # Use multi-threading to parallelize the evaluation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(evaluate_params, parameter_combinations)

    return best_params
    #######################################################################
    #                       ** End OF YOUR CODE **
    #######################################################################

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y= data.loc[:, [output_label]]
    test_portion = 1
    development_portion = 1
    training_portion = 8
    x_size = x.shape[0]
    fold_size = x_size // 10
    permutation = torch.randperm(x_size)
    development_split = permutation[:fold_size * development_portion]
    training_split = permutation[fold_size * development_portion:fold_size * (training_portion + development_portion)]
    testing_split = permutation[fold_size * (training_portion + test_portion):]

    x_train = x.iloc[training_split]
    y_train = y.iloc[training_split]
    x_dev = x.iloc[development_split]
    y_dev = y.iloc[development_split]
    x_test = x.iloc[testing_split]
    y_test = y.iloc[testing_split]

    x_final_train=pd.concat([x_train, x_dev]) #train on 
    y_final_train=pd.concat([y_train, y_dev])
    
    #uncomment to run hyperparameter search
    #hyper_params =  RegressorHyperParameterSearch(x_dev,y_dev,x_train,y_train)
    #print(hyper_params)

    #uncomment to train regressor
    regressor = Regressor(x_final_train, nb_epoch = 2300, neural_architecture=[8, 8, 8], learning_rate=0.001)
    regressor.fit(x_final_train, y_final_train)

    #save_regressor(regressor)

    # Error
    #regressor = load_regressor()
    error = regressor.score(x_test, y_test)
    print("\nRegressor error on test set: {}\n".format(error))

if __name__ == "__main__":
    example_main()

