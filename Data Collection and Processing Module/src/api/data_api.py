from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from typing import List, Dict, Any
import logging

# Setup logging
from src.utils.logging_utils import LoggingUtils
LoggingUtils.setup_logging()

# Create Flask application
app = Flask(__name__)
api = Api(app)
logger = LoggingUtils.get_logger(__name__)

class DataAPI(Resource):
    def get(self, data_type: str) -> Dict[str, Any]:
        """
        Retrieve data based on the type.

        :param data_type: Type of data to retrieve (e.g., 'satellite', 'ground', 'meteorological').
        :return: JSON response with data.
        """
        logger.info(f"Received request for data type: {data_type}")

        # Example data retrieval based on type
        # In a real application, this would involve querying a database or other data sources
        data = {
            'satellite': {'status': 'success', 'data': 'Satellite data here'},
            'ground': {'status': 'success', 'data': 'Ground data here'},
            'meteorological': {'status': 'success', 'data': 'Meteorological data here'}
        }

        if data_type in data:
            response = data[data_type]
        else:
            response = {'status': 'error', 'message': 'Data type not found'}

        logger.debug(f"Response: {response}")
        return jsonify(response)

class HealthCheck(Resource):
    def get(self) -> Dict[str, str]:
        """
        Health check endpoint to ensure the API is running.
        
        :return: JSON response indicating the status of the API.
        """
        logger.info("Health check request received")
        return jsonify({'status': 'healthy'})

# Add resources to API
api.add_resource(DataAPI, '/data/<string:data_type>')
api.add_resource(HealthCheck, '/health')

if __name__ == "__main__":
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
