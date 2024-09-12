import unittest
from unittest.mock import patch, MagicMock
from src.collectors.satellite_collector import SatelliteCollector
from src.collectors.ground_station_collector import GroundStationCollector
from src.collectors.meteorological_collector import MeteorologicalCollector
from src.collectors.resource_collector import ResourceCollector
from src.collectors.field_report_collector import FieldReportCollector
from src.collectors.drone_collector import DroneCollector

class TestCollectors(unittest.TestCase):
    def setUp(self):
        # Initialize collector instances for tests
        self.satellite_collector = SatelliteCollector()
        self.ground_station_collector = GroundStationCollector()
        self.meteorological_collector = MeteorologicalCollector()
        self.resource_collector = ResourceCollector()
        self.field_report_collector = FieldReportCollector()
        self.drone_collector = DroneCollector()

    @patch('src.collectors.satellite_collector.SomeSatelliteAPI')  # Mocking the API
    def test_satellite_collector(self, MockAPI):
        # Setup mock return value
        mock_api_instance = MockAPI.return_value
        mock_api_instance.get_data.return_value = {'status': 'success', 'data': 'Satellite data'}

        data = self.satellite_collector.collect_data()
        self.assertEqual(data, {'status': 'success', 'data': 'Satellite data'})
        MockAPI.assert_called_once()

    @patch('src.collectors.ground_station_collector.SomeGroundStationAPI')  # Mocking the API
    def test_ground_station_collector(self, MockAPI):
        # Setup mock return value
        mock_api_instance = MockAPI.return_value
        mock_api_instance.get_data.return_value = {'status': 'success', 'data': 'Ground station data'}

        data = self.ground_station_collector.collect_data()
        self.assertEqual(data, {'status': 'success', 'data': 'Ground station data'})
        MockAPI.assert_called_once()

    @patch('src.collectors.meteorological_collector.SomeMeteorologicalAPI')  # Mocking the API
    def test_meteorological_collector(self, MockAPI):
        # Setup mock return value
        mock_api_instance = MockAPI.return_value
        mock_api_instance.get_data.return_value = {'status': 'success', 'data': 'Meteorological data'}

        data = self.meteorological_collector.collect_data()
        self.assertEqual(data, {'status': 'success', 'data': 'Meteorological data'})
        MockAPI.assert_called_once()

    @patch('src.collectors.resource_collector.SomeResourceAPI')  # Mocking the API
    def test_resource_collector(self, MockAPI):
        # Setup mock return value
        mock_api_instance = MockAPI.return_value
        mock_api_instance.get_data.return_value = {'status': 'success', 'data': 'Resource data'}

        data = self.resource_collector.collect_data()
        self.assertEqual(data, {'status': 'success', 'data': 'Resource data'})
        MockAPI.assert_called_once()

    @patch('src.collectors.field_report_collector.SomeFieldReportAPI')  # Mocking the API
    def test_field_report_collector(self, MockAPI):
        # Setup mock return value
        mock_api_instance = MockAPI.return_value
        mock_api_instance.get_data.return_value = {'status': 'success', 'data': 'Field report data'}

        data = self.field_report_collector.collect_data()
        self.assertEqual(data, {'status': 'success', 'data': 'Field report data'})
        MockAPI.assert_called_once()

    @patch('src.collectors.drone_collector.SomeDroneAPI')  # Mocking the API
    def test_drone_collector(self, MockAPI):
        # Setup mock return value
        mock_api_instance = MockAPI.return_value
        mock_api_instance.get_data.return_value = {'status': 'success', 'data': 'Drone data'}

        data = self.drone_collector.collect_data()
        self.assertEqual(data, {'status': 'success', 'data': 'Drone data'})
        MockAPI.assert_called_once()

if __name__ == '__main__':
    unittest.main()
