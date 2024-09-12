from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
from typing import List, Dict, Any

class KafkaManager:
    def __init__(self, kafka_config: Dict[str, Any]):
        """
        Initialize the KafkaManager with configuration.

        :param kafka_config: Dictionary containing Kafka configuration parameters.
        """
        self.kafka_config = kafka_config
        self.producer = None
        self.consumer = None

    def create_producer(self):
        """
        Create a Kafka producer instance.
        """
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config.get('bootstrap_servers'),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Kafka producer created.")
        except KafkaError as e:
            print(f"Error creating Kafka producer: {e}")
            raise

    def create_consumer(self, topic: str, group_id: str):
        """
        Create a Kafka consumer instance.

        :param topic: Kafka topic to subscribe to.
        :param group_id: Consumer group ID.
        """
        try:
            self.consumer = KafkaConsumer(
                topic,
                group_id=group_id,
                bootstrap_servers=self.kafka_config.get('bootstrap_servers'),
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            print(f"Kafka consumer created for topic '{topic}' and group '{group_id}'.")
        except KafkaError as e:
            print(f"Error creating Kafka consumer: {e}")
            raise

    def send_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Send a message to a Kafka topic.

        :param topic: Kafka topic to send the message to.
        :param message: Message to be sent (dictionary).
        """
        if not self.producer:
            raise RuntimeError("Producer is not created. Call create_producer() first.")

        try:
            self.producer.send(topic, message)
            self.producer.flush()
            print(f"Message sent to topic '{topic}'.")
        except KafkaError as e:
            print(f"Error sending message to Kafka: {e}")
            raise

    def receive_messages(self) -> List[Dict[str, Any]]:
        """
        Receive messages from Kafka topics.

        :return: List of received messages (dictionaries).
        """
        if not self.consumer:
            raise RuntimeError("Consumer is not created. Call create_consumer() first.")

        messages = []
        for message in self.consumer:
            messages.append(message.value)
        return messages

    def close(self):
        """
        Close the Kafka producer and consumer.
        """
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
        print("Kafka producer and consumer closed.")

if __name__ == "__main__":
    # Example usage
    kafka_config = {
        'bootstrap_servers': 'localhost:9092'
    }
    
    kafka_manager = KafkaManager(kafka_config)
    
    # Create producer
    kafka_manager.create_producer()
    
    # Create consumer
    kafka_manager.create_consumer(topic='fire_reports', group_id='fire_reports_group')
    
    # Send message
    message = {
        'timestamp': '2024-08-09T12:00:00',
        'latitude': 45.4215,
        'longitude': -75.6972,
        'temperature': 25.5,
        'humidity': 60,
        'wind_speed': 10.2,
        'smoke_density': 0.1,
        'altitude': 100,
        'resource_type': 'truck',
        'report_text': 'Fire detected'
    }
    kafka_manager.send_message(topic='fire_reports', message=message)
    
    # Receive messages
    messages = kafka_manager.receive_messages()
    print(messages)
    
    # Close Kafka connections
    kafka_manager.close()
