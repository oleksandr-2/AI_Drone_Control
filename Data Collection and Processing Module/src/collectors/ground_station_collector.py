import requests
import json
import os

class GroundStationCollector:
    def __init__(self, api_key, base_url, download_dir):
        self.api_key = api_key
        self.base_url = base_url
        self.download_dir = download_dir

    def fetch_ground_station_data(self, station_id, start_time, end_time):
        """
        Fetches data from a specific ground station for a given time range.

        :param station_id: The ID of the ground station.
        :param start_time: Start time of the data collection in ISO format (YYYY-MM-DDTHH:MM:SS).
        :param end_time: End time of the data collection in ISO format (YYYY-MM-DDTHH:MM:SS).
        :return: The file path of the saved data.
        """
        endpoint = f"/data?station_id={station_id}&start_time={start_time}&end_time={end_time}&api_key={self.api_key}"
        response = requests.get(self.base_url + endpoint)

        if response.status_code == 200:
            data = response.json()
            file_name = f"ground_station_data_{station_id}_{start_time}_{end_time}.json"
            file_path = os.path.join(self.download_dir, file_name)

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            
            print(f"Data downloaded: {file_path}")
            return file_path
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return None

    def preprocess_data(self, file_path):
        """
        Preprocesses the downloaded ground station data.

        :param file_path: The path to the data file.
        :return: The path to the preprocessed data.
        """
        # Implement preprocessing logic here
        preprocessed_file_path = file_path  # For now, just returning the original file path
        print(f"Preprocessing data: {file_path}")
        return preprocessed_file_path

if __name__ == "__main__":
    api_key = "your_api_key_here"
    base_url = "https://api.groundstationprovider.com"  # Replace with actual API URL
    download_dir = "./downloaded_data"
    collector = GroundStationCollector(api_key, base_url, download_dir)

    station_id = "12345"  # Example station ID
    start_time = "2024-08-08T00:00:00"
    end_time = "2024-08-08T23:59:59"

    data_path = collector.fetch_ground_station_data(station_id, start_time, end_time)
    if data_path:
        collector.preprocess_data(data_path)
