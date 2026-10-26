# Hive 3.1.x, not 4.x: Hive Metastore 4 dropped the legacy get_table Thrift call that Spark's
# built-in Hive 2.3 client uses ("Invalid method name: 'get_table'"). Image is amd64-only.
FROM --platform=linux/amd64 apache/hive:3.1.3
# Replace the bundled 2016 Postgres driver, which cannot do SCRAM auth (Postgres 14+ default).
USER root
RUN rm /opt/hive/lib/postgresql-9.4.1208.jre7.jar
ADD --chown=hive:hive https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar /opt/hive/lib/postgresql.jar
USER hive
