import data_preparation, ARMA_Model
import time


data_preparation_object = data_preparation.DataPreparation(csv_path="number of travelers.csv", ratio=(0.7, 0.15))
coefficient_normalisation, t_train, t_validation, t_test, t_train_arma, t_validation_arma, t_test_arma, y_train_arma, y_validation_arma, y_test_arma = data_preparation_object.prepare_data_for_arma_model(p=12, q=1)
arma_model_object = ARMA_Model.ARMAModel(coefficient_normalisation,t_train, t_validation, t_test, t_train_arma, t_validation_arma, t_test_arma, y_train_arma, y_validation_arma, y_test_arma)
arma_model_object.training_gradient_descent(number_epochs=10000, learning_rate=0.000020, batch_size=2, p=12, q=1)
arma_model_object.show_forecast_of_arma_model()


