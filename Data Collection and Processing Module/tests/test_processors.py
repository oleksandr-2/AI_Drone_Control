import unittest
from unittest.mock import MagicMock
from src.processors.data_normalizer import DataNormalizer
from src.processors.data_validator import DataValidator
from src.processors.unit_converter import UnitConverter
from src.processors.anomaly_detector import AnomalyDetector

class TestProcessors(unittest.TestCase):
    
    def setUp(self):
        # Initialize processor instances for tests
        self.data_normalizer = DataNormalizer()
        self.data_validator = DataValidator()
        self.unit_converter = UnitConverter()
        self.anomaly_detector = AnomalyDetector()
    
    def test_data_normalizer(self):
        raw_data = {
            'satellite': {'temp': '300K', 'humidity': '50%'},
            'ground': {'temp': '25°C', 'humidity': '45%'}
        }
        expected_normalized_data = {
            'satellite': {'temp': 300, 'humidity': 50},
            'ground': {'temp': 298.15, 'humidity': 45}  # Assuming conversion to Kelvin
        }
        
        normalized_data = self.data_normalizer.normalize_data(raw_data)
        self.assertEqual(normalized_data, expected_normalized_data)
    
    def test_data_validator(self):
        valid_data = {'temperature': 30, 'humidity': 60}
        invalid_data = {'temperature': '30', 'humidity': '60%'}
        
        self.assertTrue(self.data_validator.validate_data(valid_data))
        self.assertFalse(self.data_validator.validate_data(invalid_data))

    def test_unit_converter(self):
        raw_data = {
            'temperature': '100F',
            'humidity': '50%'
        }
        expected_converted_data = {
            'temperature': 37.78,  # Assuming Fahrenheit to Celsius conversion
            'humidity': 50
        }
        
        converted_data = self.unit_converter.convert_units(raw_data)
        self.assertAlmostEqual(converted_data['temperature'], expected_converted_data['temperature'], places=2)
        self.assertEqual(converted_data['humidity'], expected_converted_data['humidity'])
    
    def test_anomaly_detector(self):
        data_with_anomalies = {'temperature': [30, 32, 1000], 'humidity': [60, 65, 70]}
        data_without_anomalies = {'temperature': [30, 32, 34], 'humidity': [60, 65, 70]}
        
        # Assuming anomalies are detected if values exceed a threshold
        self.assertTrue(self.anomaly_detector.detect_anomalies(data_with_anomalies))
        self.assertFalse(self.anomaly_detector.detect_anomalies(data_without_anomalies))

if __name__ == '__main__':
    unittest.main()
