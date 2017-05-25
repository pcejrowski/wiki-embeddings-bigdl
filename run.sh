#!/usr/bin/env bash

# download spark sources
# ./dev/make-distribution.sh --name custom-spark --tgz -Pyarn -Phadoop-2.7 -Dscala-2.10 -DskipTests


#SPARK_HOME="$HOME/Software/spark-2.1.1-bin-custom-spark/"
#TARGET_DIR="/home/wpitula/Studia/sem-3-mgr/wiki-embeddings-bigdl/target/scala-2.10"
#BASE_DIR="/home/wpitula/Studia/sem-3-mgr/wiki-embeddings-bigdl/datasets"

SPARK_HOME="$HOME/Private/studies/spark/"
TARGET_DIR="$HOME/Private/studies/distributed-systems/wiki-embeddings/target/scala-2.10"
BASE_DIR="$HOME/Private/studies/distributed-systems/wiki-embeddings/datasets"


${SPARK_HOME}/bin/spark-submit \
   --master "local[*]" \
   --driver-memory 8g \
   --class pl.edu.pg.eti.Main \
${TARGET_DIR}/wiki-embeddings.jar \
   --baseDir ${BASE_DIR} | tee run.log