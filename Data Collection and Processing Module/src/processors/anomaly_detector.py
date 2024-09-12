from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    def __init__(self, contamination=0.01):
        """
        Initialize the anomaly detector with a specified contamination rate.
        
        :param contamination: The proportion of anomalies in the data set.
        """
        self.model = IsolationForest(contamination=contamination)
    
    def fit(self, data):
        """
        Fit the anomaly detector to the data.

        :param data: 2D numpy array of shape (n_samples, n_features).
        """
        self.model.fit(data)
    
    def detect_anomalies(self, data):
        """
        Detect anomalies in the data.

        :param data: 2D numpy array of shape (n_samples, n_features).
        :return: List of anomaly scores.
        """
        scores = self.model.decision_function(data)
        anomalies = self.model.predict(data)
        return scores, anomalies
    
    def prepare_data(self, raw_data, feature_columns):
        """
        Prepare raw data for anomaly detection.

        :param raw_data: List of dictionaries containing data.
        :param feature_columns: List of columns to use as features.
        :return: 2D numpy array of features.
        """
        data_matrix = []
        for record in raw_data:
            row = [record.get(col, 0) for col in feature_columns]
            data_matrix.append(row)
        
        return np.array(data_matrix)

if __name__ == "__main__":
    # Example usage
    raw_data = [
        {'temperature': 25, 'humidity': 60, 'smoke_density': 0.1},
        {'temperature': 30, 'humidity': 65, 'smoke_density': 0.2},
        {'temperature': 20, 'humidity': 55, 'smoke_density': 0.05},
        {'temperature': 15, 'humidity': 70, 'smoke_density': 0.3},
        {'temperature': 1000, 'humidity': 500, 'smoke_density': 10.0}  # Outlier example
    ]
    
    feature_columns = ['temperature', 'humidity', 'smoke_density']
    
    detector = AnomalyDetector(contamination=0.2)
    data_matrix = detector.prepare_data(raw_data, feature_columns)
    detector.fit(data_matrix)
    scores, anomalies = detector.detect_anomalies(data_matrix)
    
    print("Anomaly Scores:", scores)
    print("Anomalies:", anomalies)
