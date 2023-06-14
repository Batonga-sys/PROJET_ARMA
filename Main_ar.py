import data_preparation, auto_reg_model
import time

data_preparation_object = data_preparation.DataPreparation(csv_path="number of travelers.csv", ratio=(0.7, 0.15))

coefficient_normalisation, t_train, t_validation, t_test, t_train_autoreg, t_validation_autoreg, t_test_autoreg, y_train_auto_reg, y_validation_auto_reg, y_test_auto_reg = data_preparation_object.prepare_data_for_autoregressif_model(periodicity=12)
auto_reg_model_object = auto_reg_model.AutoRegModel(coefficient_normalisation, t_train, t_validation, t_test, t_train_autoreg, t_validation_autoreg, t_test_autoreg, y_train_auto_reg, y_validation_auto_reg, y_test_auto_reg)
auto_reg_model_object.training_gradient_descent(number_epochs=8000, learning_rate=0.01, batch_size=2, periodicity=12)
auto_reg_model_object.show_forcast_of_auto_reg_model()