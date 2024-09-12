import json

class DataValidator:
    def __init__(self, schema):
        self.schema = schema

    def validate(self, data, source_type):
        """
        Validate the data from a specific source type.

        :param data: Raw data to be validated.
        :param source_type: Type of the data source (e.g., 'satellite', 'ground_station').
        :return: Validation result as a tuple (is_valid, errors).
        """
        validation_method = getattr(self, f"_validate_{source_type}_data", None)
        if validation_method is None:
            raise ValueError(f"Unknown source type: {source_type}")
        return validation_method(data)

    def _validate_satellite_data(self, data):
        # Implement validation logic for satellite data
        errors = []
        required_fields = ['timestamp', 'coordinates', 'temperature', 'smoke_density', 'image_url']

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], (str, list, float, int)):
                errors.append(f"Field '{field}' has incorrect type")

        return (len(errors) == 0, errors)

    def _validate_ground_station_data(self, data):
        # Implement validation logic for ground station data
        errors = []
        required_fields = ['timestamp', 'temperature', 'humidity', 'smoke_level']

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], (str, float, int)):
                errors.append(f"Field '{field}' has incorrect type")

        return (len(errors) == 0, errors)

    def _validate_meteorological_data(self, data):
        # Implement validation logic for meteorological data
        errors = []
        required_fields = ['timestamp', 'wind_speed', 'wind_direction', 'precipitation', 'temperature']

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], (str, float, int)):
                errors.append(f"Field '{field}' has incorrect type")

        return (len(errors) == 0, errors)

    def _validate_resource_data(self, data):
        # Implement validation logic for resource data
        errors = []
        required_fields = ['resource_id', 'resource_type', 'status', 'location']

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], (str, int)):
                errors.append(f"Field '{field}' has incorrect type")

        return (len(errors) == 0, errors)

    def _validate_field_report_data(self, data):
        # Implement validation logic for field report data
        errors = []
        required_fields = ['timestamp', 'reporter', 'report_type', 'description']

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], str):
                errors.append(f"Field '{field}' has incorrect type")

        return (len(errors) == 0, errors)

    def _validate_drone_data(self, data):
        # Implement validation logic for drone data
        errors = []
        required_fields = ['timestamp', 'drone_id', 'coordinates', 'altitude', 'camera_image_url']

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], (str, list, float, int)):
                errors.append(f"Field '{field}' has incorrect type")

        return (len(errors) == 0, errors)

if __name__ == "__main__":
    # Example usage
    schema = {}  # Define your schema if needed
    validator = DataValidator(schema)
    
    # Example raw data from different sources
    raw_data = {
        'timestamp': '2024-08-09T12:00:00',
        'coordinates': [45.0, -93.0],
        'temperature': 30,
        'smoke_density': 0.5,
        'image_url': 'http://example.com/image.jpg'
    }
    
    is_valid, errors = validator.validate(raw_data, 'satellite')
    if is_valid:
        print("Data is valid.")
    else:
        print("Data is invalid:")
        for error in errors:
            print(f"- {error}")
