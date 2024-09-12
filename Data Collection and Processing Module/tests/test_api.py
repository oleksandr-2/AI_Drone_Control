import unittest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify
from src.api.data_api import app  # Assuming 'app' is the Flask instance

class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.client.testing = True

    @patch('src.api.data_api.DataProcessor')  # Mocking the data processing class
    def test_get_processed_data(self, MockDataProcessor):
        # Setup mock return value
        mock_processor = MockDataProcessor.return_value
        mock_processor.get_processed_data.return_value = {'status': 'success', 'data': 'Processed data'}

        response = self.client.get('/api/data')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'success', 'data': 'Processed data'})

    @patch('src.api.data_api.DataProcessor')  # Mocking the data processing class
    def test_get_data_error(self, MockDataProcessor):
        # Setup mock to raise an exception
        mock_processor = MockDataProcessor.return_value
        mock_processor.get_processed_data.side_effect = Exception('Data processing error')

        response = self.client.get('/api/data')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {'status': 'error', 'message': 'Data processing error'})

    @patch('src.api.data_api.DataProcessor')  # Mocking the data processing class
    def test_post_data(self, MockDataProcessor):
        # Setup mock return value
        mock_processor = MockDataProcessor.return_value
        mock_processor.process_data.return_value = {'status': 'success', 'message': 'Data processed'}

        response = self.client.post('/api/data', json={'input': 'some data'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'success', 'message': 'Data processed'})

    @patch('src.api.data_api.DataProcessor')  # Mocking the data processing class
    def test_post_data_error(self, MockDataProcessor):
        # Setup mock to raise an exception
        mock_processor = MockDataProcessor.return_value
        mock_processor.process_data.side_effect = Exception('Data processing error')

        response = self.client.post('/api/data', json={'input': 'some data'})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {'status': 'error', 'message': 'Data processing error'})

if __name__ == '__main__':
    unittest.main()
