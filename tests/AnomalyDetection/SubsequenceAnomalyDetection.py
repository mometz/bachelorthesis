import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.ndimage import gaussian_filter1d

class SWIFD:
    def __init__(self, window_sizes=None, contamination='auto', step_factor=5, smoothing_sigma=2):
        """
        Initialisiert die SWIFD-Klasse mit den Standardparametern oder benutzerdefinierten Werten.
        
        :param window_sizes: Liste der Fenstergrößen für die Feature-Extraktion.
        :param contamination: Anteil der Daten, der als Anomalien betrachtet wird.
        :param step_factor: Faktor, der die Schrittgröße für das Gleiten des Fensters bestimmt.
        :param smoothing_sigma: Sigma-Wert für die Glättung der Anomaliedichte.
        """
        if window_sizes is None:
            window_sizes = [50, 100, 200]

        self.window_sizes = window_sizes
        self.contamination = contamination
        self.step_factor = step_factor
        self.smoothing_sigma = smoothing_sigma

    def extract_features(self, df, window_size, step_size):
        """
        Extrahiert statistische Merkmale aus einer Zeitreihe mit gleitenden Fenstern.
        """
        values = df.values
        indices = []
        features = []

        for i in range(0, len(values) - window_size, step_size):
            window = values[i : i + window_size]
            indices.append(i + window_size // 2)
            features.append([
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window)
            ])

        return np.array(features), np.array(indices)

    def IF_density(self, df):
        """
        Berechnet eine Anomalie-Dichtekarte statt nur einzelne Anomalien.
        """
        window_sizes = self.window_sizes
        contamination = self.contamination
        step_factor = self.step_factor
        smoothing_sigma = self.smoothing_sigma
        self.anomaly_density = np.zeros(len(df))

        for window_size in window_sizes:
            step_size = max(1, window_size // step_factor)
            features, indices = self.extract_features(df, window_size, step_size)

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features_scaled)
            anomalies = indices[anomaly_labels == -1]

            for idx in anomalies:
                if 0 <= idx < len(df):
                    self.anomaly_density[idx] += 1

        if smoothing_sigma == 0:
            self.smoothed_density = self.anomaly_density
        else:
            self.smoothed_density = gaussian_filter1d(self.anomaly_density, sigma=smoothing_sigma)
        
        
    def plot_anomalies(self, df):
        """
        Plottet die Zeitreihe mit farbigem Hintergrund, der die Anomaliedichte repräsentiert.
        """
        density = self.smoothed_density
        fig, ax = plt.subplots(figsize=(15, 4))

        if density.max() > 0:
            norm_density = density / density.max()
        else:
            norm_density = density

        cmap = plt.get_cmap("Reds")

        for i in range(len(df) - 1):
            color = cmap(norm_density[i])
            ax.axvspan(df.index[i], df.index[i+1], color=color, alpha=0.5)

        try:
            ax.plot(df.index, df.iloc[:, 0], label='Zeitreihe', alpha=0.8, linewidth=1.5)
        except:
            ax.plot(df.index, df, label='Zeitreihe', alpha=0.8, linewidth=1.5)
        ax.set_xlim(df.index[0], df.index[len(df)-1])
        
        ax.set_title("Anomalie-Dichtekarte")
        plt.show()
        

class GrammarViz:
    def __init__(self):
        pass

    def save_numeric_column_to_csv(self, data, column_index, filename):
        """
        Speichert eine bestimmte Spalte eines DataFrames oder NumPy-Arrays als CSV,
        wobei nur numerische Werte berücksichtigt werden.

        :param data: Pandas DataFrame oder NumPy Array
        :param column_index: Index der Spalte (für DataFrame) oder Index des Arrays, der gespeichert werden soll
        :param filename: Name der Ausgabedatei
        """

        base_folder = r'C:\Users\mmuh\Documents\Bachelorarbeit\tests\GrammarViz\\'
        filename = base_folder + filename

        if isinstance(data, pd.DataFrame):
            df_filt = data[data.apply(self.contains_number, axis=1)]
            column_data = df_filt.iloc[:, column_index]
            numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                column_data = data
            else:
                column_data = data[:, column_index]
        else:
            raise ValueError("Eingabedaten müssen entweder ein Pandas DataFrame oder ein NumPy Array sein.")
        
        numeric_data_str = numeric_data.apply(lambda x: f'{x:.10e}' if isinstance(x, float) else str(x))
        numeric_data_str.to_csv(filename, header=False, index=False)
        print(f"Gefilterte CSV wurde gespeichert als: {filename}")
        
    def contains_number(self, row):
        for value in row:
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                continue
        return False

    def filePrep(self, data, column_index, filename):
        self.save_numeric_column_to_csv(data, column_index, filename)
        
    def startGUI(self):
        ...