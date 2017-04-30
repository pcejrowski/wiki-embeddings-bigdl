#!/usr/bin/env bash

SPARK_HOME="/home/pcejrowski/Software/spark-1.5.2-bin-hadoop2.6/"
BIGDL_HOME="/home/pcejrowski/Private/studies/distributed-systems/BigDL/"
TARGET_HOME="/home/pcejrowski/Private/studies/distributed-systems/wiki-embeddings/target/scala-2.10"
BASE_DIR="/home/pcejrowski/Private/studies/distributed-systems/wiki-embeddings/datasets"

${BIGDL_HOME}/scripts/bigdl.sh -- ${SPARK_HOME}/bin/spark-submit \
   --master "local[4]" \
   --driver-memory 8g \
   --class pl.edu.pg.eti.Main \
${TARGET_HOME}/wiki-embeddings.jar \
   --baseDir ${BASE_DIR} | tee run.log