import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np

class DataPreparation:

	"""Cette classe me permet de gérer le jeu de données"""
	def __init__(self, csv_path, ratio):

		self.number_of_travelers_df = pd.read_csv(csv_path, sep=",")
		self.dataset_length = len(self.number_of_travelers_df)


		self.ratio = ratio
		self.index_split_1 = int(self.dataset_length * self.ratio[0])
		self.index_split_2 = int(self.dataset_length * (self.ratio[0]+ self.ratio[1]))

	
	def prepare_data_for_autoregressif_model(self, periodicity):

		coefficient_normalisation = self.number_of_travelers_df["passengers"].values.max()
		self.number_of_travelers_df["passengers"] /= coefficient_normalisation


		self.number_of_travelers_df["index_time"] = np.array([index_date for index_date in range(0, self.dataset_length)])

		train_dataset_df = self.number_of_travelers_df.loc[ : self.index_split_1-1]
		validation_dataset_df = self.number_of_travelers_df.loc[self.index_split_1 : self.index_split_2-1]
		test_dataset_df = self.number_of_travelers_df.loc[self.index_split_2 : ]



		t_train = train_dataset_df["index_time"].values[periodicity:]
		t_validation = validation_dataset_df["index_time"].values[periodicity:]
		t_test = test_dataset_df["index_time"].values[periodicity:]


		t_train_autoreg  = [] # liste de feature (feature une liste qui contient les p précédente valeur)
		for index_feature in range(0, len(train_dataset_df)-periodicity):
			t_train_autoreg.append(train_dataset_df["passengers"].values[index_feature : index_feature+periodicity])


		t_validation_autoreg = []  # liste de feature (feature une liste qui contient les p précédente valeur)
		for index_feature in range(0, len(validation_dataset_df) - periodicity):
			t_validation_autoreg.append(validation_dataset_df["passengers"].values[index_feature: index_feature+periodicity])

		t_test_autoreg = []  # liste de feature (feature une liste qui contient les p précédente valeur)
		for index_feature in range(0, len(test_dataset_df) - periodicity):
			t_test_autoreg.append(test_dataset_df["passengers"].values[index_feature: index_feature+periodicity])

		y_train_auto_reg = train_dataset_df["passengers"].values[periodicity:]
		y_validation_auto_reg = validation_dataset_df["passengers"].values[periodicity:]
		y_test_auto_reg = test_dataset_df["passengers"].values[periodicity:]


		return coefficient_normalisation, t_train, t_validation, t_test, t_train_autoreg, t_validation_autoreg, t_test_autoreg, y_train_auto_reg, y_validation_auto_reg, y_test_auto_reg



	def prepare_data_for_arma_model(self, p, q):
		coefficient_normalisation = self.number_of_travelers_df["passengers"].values.max()
		self.number_of_travelers_df["passengers"] /= coefficient_normalisation

		self.number_of_travelers_df["index_time"] = np.array([index_date for index_date in range(0, self.dataset_length)])

		train_dataset_df = self.number_of_travelers_df.loc[:self.index_split_1 - 1]
		validation_dataset_df = self.number_of_travelers_df.loc[self.index_split_1:self.index_split_2 - 1]
		test_dataset_df = self.number_of_travelers_df.loc[self.index_split_2:]

		t_train = train_dataset_df["index_time"].values[p + q:]
		t_validation = validation_dataset_df["index_time"].values[p + q:]
		t_test = test_dataset_df["index_time"].values[p + q:]

		t_train_arma = []  # liste de feature (feature est une liste qui contient les p précédentes valeurs)
		for index_feature in range(0, len(train_dataset_df) - p - q):
			t_train_arma.append(train_dataset_df["passengers"].values[index_feature: index_feature + p])

		t_validation_arma = []  # liste de feature (feature est une liste qui contient les p précédentes valeurs)
		for index_feature in range(0, len(validation_dataset_df) - p - q):
			t_validation_arma.append(validation_dataset_df["passengers"].values[index_feature: index_feature + p])

		t_test_arma = []  # liste de feature (feature est une liste qui contient les p précédentes valeurs)
		for index_feature in range(0, len(test_dataset_df) - p - q):
			t_test_arma.append(test_dataset_df["passengers"].values[index_feature: index_feature + p])


		y_train_arma = train_dataset_df["passengers"].values[p + q:]
		y_validation_arma = validation_dataset_df["passengers"].values[p + q:]
		y_test_arma = test_dataset_df["passengers"].values[p + q:]
		
		return (
        coefficient_normalisation, t_train, t_validation, t_test, t_train_arma,
        t_validation_arma, t_test_arma, y_train_arma, y_validation_arma, y_test_arma
    )

	
	def show_dataset(self):
		plt.figure(figsize=(15, 6))
		seaborn.scatterplot(x="index_time", y= "passengers", data=self.number_of_travelers_df)
		plt.show()


		return None









  

   