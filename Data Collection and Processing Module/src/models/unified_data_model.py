from typing import Dict, Any
import pandas as pd
import numpy as np

class UnifiedDataModel:
    def __init__(self):
        # Define default schema for the unified data model
        self.schema = {
            'timestamp': 'datetime',
            'latitude': 'float',
            'longitude': 'float',
            'temperature': 'float',
            'humidity': 'float',
            'wind_speed': 'float',
            'smoke_density': 'float',
            'altitude': 'float',
            'resource_type': 'str',
            'report_text': 'str'
        }

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate the data against the unified schema.

        :param data: Dictionary containing data to be validated.
        :return: True if data is valid, False otherwise.
        """
        for key, value_type in self.schema.items():
            if key in data:
                if not isinstance(data[key], self._get_python_type(value_type)):
                    print(f"Invalid type for {key}: Expected {value_type}, got {type(data[key]).__name__}")
                    return False
            else:
                print(f"Missing required key: {key}")
                return False
        return True

    def _get_python_type(self, value_type: str):
        """
        Map string schema types to Python types.

        :param value_type: Type as string.
        :return: Corresponding Python type.
        """
        type_map = {
            'float': float,
            'int': int,
            'str': str,
            'datetime': pd.Timestamp
        }
        return type_map.get(value_type, str)

    def convert_to_dataframe(self, data_list: [Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert a list of data dictionaries to a pandas DataFrame.

        :param data_list: List of dictionaries containing data.
        :return: DataFrame with standardized columns.
        """
        if not data_list:
            return pd.DataFrame(columns=self.schema.keys())

        data_frame = pd.DataFrame(data_list)
        
        for key, value_type in self.schema.items():
            if key in data_frame.columns:
                data_frame[key] = data_frame[key].astype(self._get_python_type(value_type))
        
        return data_frame

    def prepare_data(self, raw_data: [Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare raw data for processing.

        :param raw_data: List of dictionaries containing raw data.
        :return: DataFrame with validated and standardized data.
        """
        valid_data = [item for item in raw_data if self.validate_data(item)]
        return self.convert_to_dataframe(valid_data)

if __name__ == "__main__":
    # Example usage
    raw_data = [
        {'timestamp': '2024-08-09T12:00:00', 'latitude': 45.4215, 'longitude': -75.6972, 'temperature': 25.5, 'humidity': 60, 'wind_speed': 10.2, 'smoke_density': 0.1, 'altitude': 100, 'resource_type': 'truck', 'report_text': 'Fire detected'},
        {'timestamp': '2024-08-09T12:05:00', 'latitude': 45.4216, 'longitude': -75.6973, 'temperature': 26.0, 'humidity': 62, 'wind_speed': 9.8, 'smoke_density': 0.2, 'altitude': 105, 'resource_type': 'helicopter', 'report_text': 'No significant changes'},
        {'timestamp': '2024-08-09T12:10:00', 'latitude': 45.4217, 'longitude': -75.6974, 'temperature': 27.5, 'humidity': 64, 'wind_speed': 11.0, 'smoke_density': 0.3, 'altitude': 110, 'resource_type': 'plane', 'report_text': 'Heavy smoke'}
    ]
    
    model = UnifiedDataModel()
    prepared_data = model.prepare_data(raw_data)
    print(prepared_data)
