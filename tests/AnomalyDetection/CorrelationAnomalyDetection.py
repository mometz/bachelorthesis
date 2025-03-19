import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

class MDSWIFD:
    def __init__(self, windows: np.ndarray, contamination: float | str = 'auto', step_factor: int = 5, smoothing_sigma: int = 2):
        self.windows = windows
        self.contamination = contamination
        self.step_factor = step_factor
        self.smoothing_sigma = smoothing_sigma
        
    def compute_mahalanobis_distance(self, df):
        """
        Berechnet die Mahalanobis-Distanz für eine multivariate oder univariate Zeitreihe.
        
        :param df: DataFrame mit numerischen Spalten (keine Zeitstempel)
        :return: np.array mit Mahalanobis-Distanzen
        """
        data = df.values  # Konvertiere DataFrame zu NumPy-Array
        mean = np.mean(data, axis=0)
        
        if data.shape[1] == 1:  # Univariater Fall
            std_dev = np.std(data, axis=0)
            distances = np.abs((data - mean) / std_dev)  # Z-Score
        else:  # Multivariater Fall
            cov_matrix = np.cov(data, rowvar=False)
            inv_cov_matrix = inv(cov_matrix)
            distances = np.array([mahalanobis(x, mean, inv_cov_matrix) for x in data])

        return distances
    
    def extract_features(self, df, window_size, step_size):
        """
        Extrahiert statistische Merkmale aus einer Zeitreihe mit gleitenden Fenstern.
        """
        values = df.values
        indices = []
        features = []

        for i in range(0, len(values) - window_size, step_size):
            window = values[i : i + window_size]
            indices.append(i + window_size // 2)  # Mittelpunkt des Fensters
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
        column = 'value'
        anomaly_density = np.zeros(len(df))

        for window_size in self.windows:
            step_size = max(1, window_size // self.step_factor)
            features, indices = self.extract_features(df, window_size, step_size)

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features_scaled)
            anomalies = indices[anomaly_labels == -1]

            # Erhöhe die Dichtewerte für erkannte Anomalien
            for idx in anomalies:
                if 0 <= idx < len(df):
                    anomaly_density[idx] += 1

        # Glätten der Anomaliedichte für bessere Visualisierung
        smoothed_density = gaussian_filter1d(anomaly_density, sigma=self.smoothing_sigma)
        
        return smoothed_density, features

    def IF_density_mahalanobis(self, df):
        """
        Wendet SWIFD auf der Mahalanobis-Distanz-Zeitreihe an.
        """
        # Mahalanobis-Distanz berechnen
        if df.shape[1] == 1:  # Univariater Fall
            self.mahalanobis_series = self.compute_mahalanobis_distance(df)
            df_mahalanobis = pd.DataFrame({'value': self.mahalanobis_series.flatten()}, index=df.index)
        else:
            self.mahalanobis_series = self.compute_mahalanobis_distance(df)
            df_mahalanobis = pd.DataFrame({'value': self.mahalanobis_series}, index=df.index)
        # Standardmäßige SWIFD-Anomalieerkennung mit der Distanz-Zeitreihe
        self.anomaly_density, _ = self.IF_density(df_mahalanobis)
        
    def plot_anomaly_background(self, df):
        """
        Plottet die Zeitreihe mit farbigem Hintergrund, der die Anomaliedichte repräsentiert.
        """
        density = self.anomaly_density
        fig, ax = plt.subplots(figsize=(15, 4))

        # Falls `density` nur Nullen enthält, wird die Skalierung vermieden
        if density.max() > 0:
            norm_density = density / density.max()  # Normalisieren für Farbintensität
        else:
            norm_density = density  # Falls alles 0 ist, keine Normierung

        cmap = plt.get_cmap("Reds")  # Richtiger Zugriff auf die Colormap

        # Hintergrund einfärben basierend auf der Anomalie-Dichte
        for i in range(len(df) - 1):
            color = cmap(norm_density[i])  # Farbe aus der Reds-Skala holen
            ax.axvspan(df.index[i], df.index[i+1], color=color, alpha=0.5)  # Farbbereich zwischen Zeitpunkten

        # Zeitreihe darstellen
        ax.plot(df.index, df, label='Zeitreihe', alpha=0.8, linewidth=1.5)

        ax.set_title("Anomalie-Heatmap als Hintergrund")
        ax.set_ylabel("Wert")
        ax.set_xlabel("Zeit")
        ax.set_xlim(df.index[0], df.index[-1])
        ax.legend()
        plt.show()