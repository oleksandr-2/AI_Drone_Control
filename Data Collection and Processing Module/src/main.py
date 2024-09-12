from src.collectors.satellite_collector import SatelliteCollector
from src.collectors.ground_station_collector import GroundStationCollector
from src.collectors.meteorological_collector import MeteorologicalCollector
from src.collectors.resource_collector import ResourceCollector
from src.collectors.field_report_collector import FieldReportCollector
from src.collectors.drone_collector import DroneCollector
from src.processors.data_normalizer import DataNormalizer
from src.processors.data_validator import DataValidator
from src.processors.unit_converter import UnitConverter
from src.processors.anomaly_detector import AnomalyDetector
from src.utils.logging_utils import LoggingUtils

# Setup logging
LoggingUtils.setup_logging()

def main():
    # Initialize collectors
    satellite_collector = SatelliteCollector()
    ground_station_collector = GroundStationCollector()
    meteorological_collector = MeteorologicalCollector()
    resource_collector = ResourceCollector()
    field_report_collector = FieldReportCollector()
    drone_collector = DroneCollector()
    
    # Collect data
    satellite_data = satellite_collector.collect_data()
    ground_station_data = ground_station_collector.collect_data()
    meteorological_data = meteorological_collector.collect_data()
    resource_data = resource_collector.collect_data()
    field_report_data = field_report_collector.collect_data()
    drone_data = drone_collector.collect_data()

    # Initialize processors
    data_normalizer = DataNormalizer()
    data_validator = DataValidator()
    unit_converter = UnitConverter()
    anomaly_detector = AnomalyDetector()

    # Process data
    all_data = {
        'satellite': satellite_data,
        'ground': ground_station_data,
        'meteorological': meteorological_data,
        'resource': resource_data,
        'field_report': field_report_data,
        'drone': drone_data
    }

    # Normalize, validate, convert units, and detect anomalies
    normalized_data = data_normalizer.normalize_data(all_data)
    validated_data = data_validator.validate_data(normalized_data)
    converted_data = unit_converter.convert_units(validated_data)
    final_data = anomaly_detector.detect_anomalies(converted_data)

    # Logging
    logger = LoggingUtils.get_logger(__name__)
    logger.info("Data collection and processing completed.")
    logger.debug(f"Final processed data: {final_data}")

if __name__ == "__main__":
    main()
