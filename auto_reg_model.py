import random
import tqdm
import matplotlib.pyplot as plt
from more_itertools import chunked
import numpy
import time


class AutoRegModel:
	def __init__(self, coefficient_normalisation, t_train, t_validation, t_test, t_train_autoreg, t_validation_autoreg, t_test_autoreg, y_train_auto_reg, y_validation_auto_reg, y_test_auto_reg):
		
		self.coefficient_normalisation = coefficient_normalisation

		self.t_train = t_train
		self.t_validation = t_validation
		self.t_test = t_test

		
		self.t_train_autoreg = t_train_autoreg
		self.t_validation_autoreg = t_validation_autoreg
		self.t_test_autoreg = t_test_autoreg
		self.y_train_auto_reg = y_train_auto_reg
		self.y_validation_auto_reg = y_validation_auto_reg
		self.y_test_auto_reg = y_test_auto_reg



	def training_gradient_descent(self, number_epochs, learning_rate, batch_size, periodicity):
		train_length = len(self.t_train_autoreg)
		validation_length = len(self.t_validation_autoreg)

		self.parametres = numpy.zeros(shape=(periodicity + 1)) # les premiers termes sont les ap coefficient, et le dernier terme c'est le biais !!!

		# auto_reg_model_forecast = lambda feature : (self.parametres*feature.T)[0] # bugué # produit d'a
		auto_reg_model_forecast = lambda feature : numpy.matmul(self.parametres[:-1], feature) + self.parametres[-1]


		loss_function_mse_train = lambda predictions: sum([(predictions[index_feature] - self.y_train_auto_reg[index_feature])**2 for index_feature in range(0, train_length)]) / (2*train_length)
		loss_function_mse_validation = lambda predictions: sum([(predictions[index_feature] - self.y_validation_auto_reg[index_feature])**2 for index_feature in range(0, validation_length)]) / (2*validation_length)
		
		list_epochs = []
		list_mse_loss_values_for_train_per_epoch = []
		list_mse_loss_values_for_validation_per_epoch = []


		t_train_autoreg_batches = numpy.array(list(chunked(self.t_train_autoreg, batch_size)))
		y_train_auto_reg_batches = numpy.array(list(chunked(self.y_train_auto_reg, batch_size)))

		number_of_batches = len(t_train_autoreg_batches)

		for epoch in range(0, number_epochs):

			for index_batch in range(0, number_of_batches):
				number_of_element_current_batch = len(t_train_autoreg_batches[index_batch])

				current_batch_train_predictions = list(map(auto_reg_model_forecast, t_train_autoreg_batches[index_batch]))

	
				for index_periodicity in range(0, periodicity):
					dloss_dak =  (1/number_of_element_current_batch) * sum([t_train_autoreg_batches[index_batch][index_feature][index_periodicity] * (current_batch_train_predictions[index_feature] - y_train_auto_reg_batches[index_batch][index_feature])\
																																		for index_feature in range(0, number_of_element_current_batch)])
					self.parametres[index_periodicity] -= learning_rate * dloss_dak


				dloss_db = (1/number_of_element_current_batch) * sum([(current_batch_train_predictions[index_feature] - y_train_auto_reg_batches[index_batch][index_feature])\
																										for index_feature in range(0, number_of_element_current_batch)])
				self.parametres[-1] -= learning_rate * dloss_db


			train_predictions_current_epoch = list(map(auto_reg_model_forecast, self.t_train_autoreg))
			validation_predictions_current_epoch = list(map(auto_reg_model_forecast, self.t_validation_autoreg))

			loss_train_current_epoch  = loss_function_mse_train(train_predictions_current_epoch)
			loss_validation_current_epoch = loss_function_mse_validation(validation_predictions_current_epoch)

			list_epochs.append(epoch)
			list_mse_loss_values_for_train_per_epoch.append(loss_train_current_epoch)
			list_mse_loss_values_for_validation_per_epoch.append(loss_validation_current_epoch)

			if epoch % 1000 ==0 and epoch!=0:
				print(f"epoch : {epoch}")
				print(loss_train_current_epoch, loss_validation_current_epoch)
				print(self.parametres)
				print("#"*20)
				

		# print(f"la valeur de ma mse est : {loss_function_mse_train(train_predictions_current_epoch)}")
		plt.figure(figsize=(15, 6))
		plt.plot(list_epochs[100:], list_mse_loss_values_for_train_per_epoch[100:], "b")
		plt.plot(list_epochs[100:], list_mse_loss_values_for_validation_per_epoch[100:], "r")
		plt.show()
		return None


	def show_forcast_of_auto_reg_model(self):
		if not hasattr(self, "parametres"):
			print("Error : you must first train your model !!")

		else:
			auto_reg_model_forecast = lambda feature : numpy.matmul(self.parametres[:-1], feature) + self.parametres[-1]

			train_predictions = list(map(auto_reg_model_forecast, self.t_train_autoreg))

			validation_predictions = list(map(auto_reg_model_forecast, self.t_validation_autoreg))
			test_predictions = list(map(auto_reg_model_forecast, self.t_test_autoreg))


			plt.figure(figsize=(15, 6))
			# train base
			plt.plot(self.t_train, self.y_train_auto_reg, "o", color="cyan")
			plt.plot(self.t_train, train_predictions, "o", color="blue")

			# train validation
			plt.plot(self.t_validation, self.y_validation_auto_reg, "o", color="cyan")
			plt.plot(self.t_validation, validation_predictions, "o", color="blue")
			#test base

			plt.plot(self.t_test, self.y_test_auto_reg, "o", color="orange")
			plt.plot(self.t_test, test_predictions, "o", color="red")

			plt.show()

			return None