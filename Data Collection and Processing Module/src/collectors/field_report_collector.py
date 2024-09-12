import requests
import json
import os

class FieldReportCollector:
    def __init__(self, api_key, base_url, download_dir):
        self.api_key = api_key
        self.base_url = base_url
        self.download_dir = download_dir

    def fetch_field_reports(self, report_id=None, start_time=None, end_time=None):
        """
        Fetches field reports based on optional report ID and/or time range.

        :param report_id: The ID of the specific report to fetch (optional).
        :param start_time: Start time of the data collection in ISO format (YYYY-MM-DDTHH:MM:SS) (optional).
        :param end_time: End time of the data collection in ISO format (YYYY-MM-DDTHH:MM:SS) (optional).
        :return: The file path of the saved data.
        """
        if report_id:
            endpoint = f"/field_reports/{report_id}?api_key={self.api_key}"
        elif start_time and end_time:
            endpoint = f"/field_reports?start_time={start_time}&end_time={end_time}&api_key={self.api_key}"
        else:
            raise ValueError("Either report_id or both start_time and end_time must be provided.")
        
        response = requests.get(self.base_url + endpoint)

        if response.status_code == 200:
            data = response.json()
            file_name = f"field_reports_{start_time}_{end_time}.json" if not report_id else f"field_report_{report_id}.json"
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
        Preprocesses the downloaded field report data.

        :param file_path: The path to the data file.
        :return: The path to the preprocessed data.
        """
        # Implement preprocessing logic here
        preprocessed_file_path = file_path  # For now, just returning the original file path
        print(f"Preprocessing data: {file_path}")
        return preprocessed_file_path

if __name__ == "__main__":
    api_key = "your_api_key_here"
    base_url = "https://api.fieldreportprovider.com"  # Replace with actual API URL
    download_dir = "./downloaded_data"
    collector = FieldReportCollector(api_key, base_url, download_dir)

    report_id = None  # Set to a specific report ID if needed
    start_time = "2024-08-08T00:00:00"
    end_time = "2024-08-08T23:59:59"

    data_path = collector.fetch_field_reports(report_id, start_time, end_time)
    if data_path:
        collector.preprocess_data(data_path)
