import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv('number of travelers.csv')
# Assurez-vous que votre fichier CSV est bien formaté avec une colonne de dates/temps et une colonne de valeurs

# Conversion de la colonne de dates en index temporel
data['month'] = pd.to_datetime(data['month'])
data.set_index('month', inplace=True)

# Calcul de l'autocorrélation
autocorr = sm.graphics.tsa.plot_acf(data['passengers'].values, lags=30)
plt.title("Autocorrélation")
plt.show()

