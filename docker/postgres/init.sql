-- One Postgres instance backs both the Airflow metadata DB and the Hive Metastore.
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;
CREATE USER hive WITH PASSWORD 'hive';
CREATE DATABASE metastore OWNER hive;
