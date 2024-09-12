import requests
import json
import os

class ResourceCollector:
    def __init__(self, api_key, base_url, download_dir):
        self.api_key = api_key
        self.base_url = base_url
        self.download_dir = download_dir

    def fetch_resource_data(self, resource_type, region):
        """
        Fetches data about resources of a specific type in a given region.

        :param resource_type: The type of resource to fetch (e.g., "fire_department", "equipment", "reservoir").
        :param region: The geographical region for the data (e.g., "region_name").
        :return: The file path of the saved data.
        """
        endpoint = f"/resources?type={resource_type}&region={region}&api_key={self.api_key}"
        response = requests.get(self.base_url + endpoint)

        if response.status_code == 200:
            data = response.json()
            file_name = f"resource_data_{resource_type}_{region}.json"
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
        Preprocesses the downloaded resource data.

        :param file_path: The path to the data file.
        :return: The path to the preprocessed data.
        """
        # Implement preprocessing logic here
        preprocessed_file_path = file_path  # For now, just returning the original file path
        print(f"Preprocessing data: {file_path}")
        return preprocessed_file_path

if __name__ == "__main__":
    api_key = "your_api_key_here"
    base_url = "https://api.resourceprovider.com"  # Replace with actual API URL
    download_dir = "./downloaded_data"
    collector = ResourceCollector(api_key, base_url, download_dir)

    resource_type = "fire_department"  # Example resource type
    region = "Los_Angeles"  # Example region

    data_path = collector.fetch_resource_data(resource_type, region)
    if data_path:
        collector.preprocess_data(data_path)
