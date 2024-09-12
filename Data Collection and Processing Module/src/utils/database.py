import psycopg2
from psycopg2 import sql
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the DatabaseManager with configuration.

        :param db_config: Dictionary containing database configuration parameters.
        """
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        self.connect()

    def connect(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        try:
            self.connection = psycopg2.connect(
                dbname=self.db_config.get('dbname'),
                user=self.db_config.get('user'),
                password=self.db_config.get('password'),
                host=self.db_config.get('host'),
                port=self.db_config.get('port')
            )
            self.cursor = self.connection.cursor()
            print("Database connection established.")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        """
        Close the connection to the PostgreSQL database.
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Database connection closed.")

    def execute_query(self, query: str, params: Tuple = ()) -> None:
        """
        Execute a single SQL query.

        :param query: SQL query to be executed.
        :param params: Tuple of parameters to pass to the query.
        """
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()
            raise

    def fetch_all(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and fetch all results.

        :param query: SQL query to be executed.
        :param params: Tuple of parameters to pass to the query.
        :return: List of dictionaries containing the query results.
        """
        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except psycopg2.Error as e:
            print(f"Error fetching data: {e}")
            raise

    def create_table(self, table_name: str, columns: Dict[str, str]) -> None:
        """
        Create a new table in the database.

        :param table_name: Name of the table to be created.
        :param columns: Dictionary mapping column names to data types.
        """
        column_defs = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
        query = sql.SQL("CREATE TABLE IF NOT EXISTS {table} ({columns})").format(
            table=sql.Identifier(table_name),
            columns=sql.SQL(column_defs)
        )
        self.execute_query(query.as_string(self.connection))

    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """
        Insert data into a table.

        :param table_name: Name of the table.
        :param data: List of dictionaries containing data to be inserted.
        """
        if not data:
            return

        columns = data[0].keys()
        column_names = ', '.join(columns)
        values_placeholder = ', '.join([f"%({col})s" for col in columns])
        query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
            table=sql.Identifier(table_name),
            columns=sql.SQL(column_names),
            values=sql.SQL(values_placeholder)
        )
        try:
            self.cursor.executemany(query.as_string(self.connection), data)
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error inserting data: {e}")
            self.connection.rollback()
            raise

if __name__ == "__main__":
    # Example usage
    db_config = {
        'dbname': 'wildfire_db',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'localhost',
        'port': '5432'
    }
    
    db = DatabaseManager(db_config)
    
    # Create a table
    db.create_table('fire_reports', {
        'id': 'SERIAL PRIMARY KEY',
        'timestamp': 'TIMESTAMP',
        'latitude': 'FLOAT',
        'longitude': 'FLOAT',
        'temperature': 'FLOAT',
        'humidity': 'FLOAT',
        'wind_speed': 'FLOAT',
        'smoke_density': 'FLOAT',
        'altitude': 'FLOAT',
        'resource_type': 'VARCHAR(255)',
        'report_text': 'TEXT'
    })
    
    # Insert data
    data = [
        {'timestamp': '2024-08-09T12:00:00', 'latitude': 45.4215, 'longitude': -75.6972, 'temperature': 25.5, 'humidity': 60, 'wind_speed': 10.2, 'smoke_density': 0.1, 'altitude': 100, 'resource_type': 'truck', 'report_text': 'Fire detected'},
        {'timestamp': '2024-08-09T12:05:00', 'latitude': 45.4216, 'longitude': -75.6973, 'temperature': 26.0, 'humidity': 62, 'wind_speed': 9.8, 'smoke_density': 0.2, 'altitude': 105, 'resource_type': 'helicopter', 'report_text': 'No significant changes'}
    ]
    
    db.insert_data('fire_reports', data)
    
    # Fetch data
    results = db.fetch_all("SELECT * FROM fire_reports")
    print(results)
    
    db.disconnect()
