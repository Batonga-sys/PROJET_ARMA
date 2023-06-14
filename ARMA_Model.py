import random
import tqdm
import matplotlib.pyplot as plt
from more_itertools import chunked
import numpy as np
import time


# t_train_arma, t_validation_arma, t_test_arma : ensembles de données temporelles d'entraînement, de val et de test 
# y_train_arma, y_validation_arma, y_test_arma : ensembles de données cibles d'entraînement, de val et de test 
class ARMAModel:
    def __init__(self, coefficient_normalisation, t_train, t_validation, t_test, t_train_arma, t_validation_arma, t_test_arma, y_train_arma, y_validation_arma, y_test_arma):
        self.coefficient_normalisation = coefficient_normalisation
        self.t_train = t_train
        self.t_validation = t_validation
        self.t_test = t_test
        self.t_train_arma = t_train_arma
        self.t_validation_arma = t_validation_arma
        self.t_test_arma = t_test_arma
        self.y_train_arma = y_train_arma
        self.y_validation_arma = y_validation_arma
        self.y_test_arma = y_test_arma

    def training_gradient_descent(self, number_epochs, learning_rate, batch_size, p, q):
        train_length = len(self.t_train_arma)
        validation_length = len(self.t_validation_arma)
        self.parameters = np.zeros(p + q + 1)  # Les premiers termes (p) sont les coefficients AR, les termes suivants (q) sont les coefficients MA, et le dernier terme est le biais.


 # calcul les prédictions du modèle ARMA
        def arma_model_forecast(feature):
            ar_term = np.dot(self.parameters[:p], feature)
            ma_term = np.dot(self.parameters[p:p+q], self.residuals[-q:])
            return ar_term + ma_term + self.parameters[-1]

# MSE entre les prédictions et les valeurs réelles de l'ensemble d'entraînement
        loss_function_mse_train = lambda predictions: np.mean((predictions - self.y_train_arma)**2) / 2

# MSE entre les prédictions et les valeurs réelles de l'ensemble de validation
        loss_function_mse_validation = lambda predictions: np.mean((predictions - self.y_validation_arma)**2) / 2

# Résidus mis à jour à chque itération
        self.residuals = np.zeros(max(p, q))

# Des listes vides pr stocker les informations d'entraînement du modèle
        list_epochs = []
        list_mse_loss_values_for_train_per_epoch = []
        list_mse_loss_values_for_validation_per_epoch = []

# Division des données d'entrainement en lots (batches)
        t_train_arma_batches = np.array(list(chunked(self.t_train_arma, batch_size)), dtype=object)
        y_train_arma_batches = np.array(list(chunked(self.y_train_arma, batch_size)), dtype=object)

        number_of_batches = len(t_train_arma_batches) # Stockage du nombre de batches crées 

        for epoch in range(number_epochs):
            for index_batch in range(number_of_batches):
                current_batch_train_predictions = list(map(arma_model_forecast, t_train_arma_batches[index_batch]))

# Mise à jour des coefficients AR (p) en utilisant la méthode de descente de gradient
                for index_periodicity in range(p):
                    dloss_dak = (1 / batch_size) * np.sum(np.array(t_train_arma_batches[index_batch])[:, index_periodicity] * (np.array(current_batch_train_predictions) - np.array(y_train_arma_batches[index_batch])))
                    self.parameters[index_periodicity] -= learning_rate * dloss_dak


# Mise à jour des coefficients MA (q) en utilisant la méthode de descente de gradient
                for index_periodicity in range(q):
                    dloss_dbk = (1 / batch_size) * np.sum(self.residuals[-q+index_periodicity] * (np.array(current_batch_train_predictions) - y_train_arma_batches[index_batch]))
                    self.parameters[p + index_periodicity] -= learning_rate * dloss_dbk


#  Mise à jour du biais, Le calcul de la dérivée de la fonction de perte par rapport au biais 
                dloss_db = (1 / batch_size) * np.sum(np.array(current_batch_train_predictions) - y_train_arma_batches[index_batch])
                self.parameters[-1] -= learning_rate * dloss_db


# Calcul des prédictions sur le train & val 
            train_predictions_current_epoch = list(map(arma_model_forecast, self.t_train_arma))
            validation_predictions_current_epoch = list(map(arma_model_forecast, self.t_validation_arma))
            
            self.residuals = np.array(self.y_train_arma) - np.array(train_predictions_current_epoch) # résidus

# Calcul de la perte MSE (Mean Squared Error) sur le train et la val en utilisant les fonctions de perte
            loss_train_current_epoch = loss_function_mse_train(train_predictions_current_epoch)
            loss_validation_current_epoch = loss_function_mse_validation(validation_predictions_current_epoch)

            list_epochs.append(epoch)
            list_mse_loss_values_for_train_per_epoch.append(loss_train_current_epoch)
            list_mse_loss_values_for_validation_per_epoch.append(loss_validation_current_epoch)

            if epoch % 1000 == 0 and epoch != 0:
                print(f"epoch : {epoch}")
                print(loss_train_current_epoch, loss_validation_current_epoch)
                print(self.parameters)
                print("#" * 20)

        plt.figure(figsize=(15, 6))
        plt.plot(list_epochs[100:], list_mse_loss_values_for_train_per_epoch[100:], "b") # Tracé de la courbe de perte MSE de train
        plt.plot(list_epochs[100:], list_mse_loss_values_for_validation_per_epoch[100:], "r") #Tracé de la courbe de perte MSE de val
        plt.show()

        return None

    def show_forecast_of_arma_model(self):
        if not hasattr(self, "parameters"):
            print("Error: you must first train your model!")
        else:
            p = len(self.parameters) - 2
            q = 1
            arma_model_forecast = lambda feature: np.dot(self.parameters[:p], feature) + np.dot(self.parameters[p:p+q], self.residuals[-q:]) + self.parameters[-1]

            train_predictions = list(map(arma_model_forecast, self.t_train_arma))
            validation_predictions = list(map(arma_model_forecast, self.t_validation_arma))
            test_predictions = list(map(arma_model_forecast, self.t_test_arma))

            plt.figure(figsize=(15, 6))
            # Train base
            plt.plot(self.t_train, self.y_train_arma, "o", color="cyan")
            plt.plot(self.t_train, train_predictions, "o", color="blue")

            # Validation base
            plt.plot(self.t_validation, self.y_validation_arma, "o", color="cyan")
            plt.plot(self.t_validation, validation_predictions, "o", color="blue")

            # Test base
            plt.plot(self.t_test, self.y_test_arma, "o", color="orange")
            plt.plot(self.t_test, test_predictions, "o", color="red")

            plt.show()

        return None
