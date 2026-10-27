FROM apache/airflow:2.10.5-python3.12

USER root
# Spark 4 needs Java 17+.
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*
RUN ln -s "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /opt/java
ENV JAVA_HOME=/opt/java

USER airflow
# numba runs the local sample simulation. pyarrow is pinned to the same version that writes the
# landing files: the image's pyarrow 19 cannot read pyarrow 20 Parquet size statistics
# ("Repetition level histogram size mismatch").
RUN pip install --no-cache-dir "pyspark==4.0.1" "numba==0.61.2" "pyarrow==20.0.0"
