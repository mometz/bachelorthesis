import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union

class SWZScore:
    def __init__(self, windows: np.ndarray, threshold: float = 3.0):
        self.windows: np.ndarray = windows
        self.threshold: float = threshold
        self.df: Union[pd.DataFrame, None] = None
        self.anomaly_scores: Union[pd.Series, None] = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, np.ndarray):  
            self.df = pd.DataFrame(data, columns=['value'])
        elif isinstance(data, pd.DataFrame):  
            self.df = data.copy()
        else:
            raise TypeError("data muss entweder ein NumPy-Array oder ein Pandas DataFrame sein")
        
        # Initialisiere den Score-Accumulator
        self.df['anomaly_score'] = 0.0
        
        for window_size in self.windows:
            rolling_mean = self.df['value'].rolling(window=window_size).mean()
            rolling_std = self.df['value'].rolling(window=window_size).std()
            z_score = (self.df['value'] - rolling_mean) / rolling_std
            anomaly_mask = z_score.abs() > self.threshold
            
            # Anomalie-Score aufsummieren, je öfter eine Stelle als Anomalie erkannt wird, desto höher
            self.df['anomaly_score'] += anomaly_mask.astype(float)
        
        # Normierung der Scores auf [0,1] für bessere Skalierung im Plot
        max_score = self.df['anomaly_score'].max()
        if max_score > 0:
            self.df['anomaly_score'] /= max_score
        
        return self.df

    def predict(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Das Modell muss zuerst mit `fit(data)` trainiert werden.")
        return self.df[self.df['anomaly_score'] > 0]

    def plot(self) -> None:
        if self.df is None:
            raise ValueError("Das Modell muss zuerst mit `fit(data)` trainiert werden.")

        plt.figure(figsize=(15, 4))
        plt.plot(self.df.index, self.df['value'], label='Zeitserie', linewidth=0.5, color='blue')
        
        # Filtere den DataFrame, um nur Zeilen mit anomaly_score >= 1 zu behalten
        filtered_df = self.df[self.df['anomaly_score'] == True]

        # Erstelle den Scatter-Plot nur mit den gefilterten Daten
        plt.scatter(filtered_df.index, filtered_df['value'], 
                            c='red', label='Anomalien',
                            edgecolor='k', linewidth=0.2)

        plt.title('Rolling Z-Score Anomaly Detection')
        plt.xlabel('Zeit')
        plt.ylabel('Wert')
        plt.legend()
        plt.grid()
        plt.show()


from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler

from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class HBOSDetector:
    def __init__(self, windows: np.ndarray, step_size: int = 10, contamination: float = 0.0025, n_bins='auto'):
        """
        Initialisiert die HBOS-Anomaliedetektionsklasse mit variablen Fenstergrößen.
        
        :param windows: Liste von Fenstergrößen für die Sliding Window Analyse.
        :param step_size: Schrittweite des Fensters.
        :param contamination: Erwarteter Anteil an Anomalien in den Daten.
        """
        self.windows = windows
        self.step_size = step_size
        self.contamination = contamination
        self.df = None
        self.n_bins = n_bins
    
    def fit(self, data: np.ndarray | pd.DataFrame):
        """
        Führt die HBOS-Anomalieerkennung mit Sliding Windows auf den Daten durch.
        
        :param data: NumPy-Array oder Pandas DataFrame mit einer einzelnen Spalte 'value'.
        :return: Pandas DataFrame mit Anomalien und Scores.
        """
        if isinstance(data, np.ndarray):
            self.df = pd.DataFrame(data, columns=['value'])
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise TypeError("data muss entweder ein NumPy-Array oder ein Pandas DataFrame sein")
        
        self.df['anomaly'] = 0
        self.df['score'] = 0.0
        
        for window_size in self.windows:
            anomalies = np.zeros(len(self.df))
            scores = np.zeros(len(self.df))
            
            for start in range(0, len(self.df) - window_size, self.step_size):
                end = start + window_size
                window_data = np.array(self.df['value'].iloc[start:end]).reshape(-1, 1)  # numpy array mit reshape
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(window_data)
                
                clf = HBOS(n_bins=self.n_bins, contamination=self.contamination)  # type: ignore # 'auto' durch eine Zahl ersetzen
                clf.fit(X_scaled)
                
                window_anomalies = clf.labels_
                window_scores = clf.decision_scores_
                
                anomalies[start:end] += window_anomalies
                scores[start:end] = np.maximum(scores[start:end], window_scores) # type: ignore
            
            self.df[f'anomaly_{window_size}'] = (anomalies > 0).astype(int)
            self.df[f'score_{window_size}'] = scores
        
        return self.df

    def plot(self) -> None:
        if self.df is None:
            raise ValueError("Das Modell muss zuerst mit `fit(data)` trainiert werden.")
        
        # Erstelle den Basisplot für die Werte
        plt.figure(figsize=(15, 4))
        plt.plot(self.df.index, self.df['value'], label='Wert', linewidth=0.5, color='blue')
        
        # Aggregiere die Anomalien über alle Fenstergrößen hinweg
        anomaly_mask = self.df[[f'anomaly_{w}' for w in self.windows]].any(axis=1)
        
        # Filtere die Anomalien
        filtered_df = self.df[anomaly_mask]

        # Scatter-Plot für alle Anomalien
        plt.scatter(filtered_df.index, filtered_df['value'], 
                    color='red', edgecolor='k', linewidth=0.2, label='Anomalien')

        plt.title('Anomalieerkennung mit Sliding Window über alle Fenstergrößen')
        plt.xlabel('Zeit')
        plt.ylabel('Wert')
        plt.legend()
        plt.grid()
        plt.show()
