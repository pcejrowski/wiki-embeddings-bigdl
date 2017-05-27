#!/usr/bin/env bash


whoami=$(whoami)

SPARK_HOME="$HOME/Software/spark-2.1.1-bin-custom-spark/"
if [ "${whoami}" = "wpitula" ]
then
    TARGET_DIR="$HOME/Studia/sem-3-mgr/wiki-embeddings-bigdl/target/scala-2.10"
    BASE_DIR="$HOME/Studia/sem-3-mgr/wiki-embeddings-bigdl/datasets"
elif [ "${whoami}" = "pcejrowski" ]
then
    TARGET_DIR="$HOME/Private/studies/distributed-systems/wiki-embeddings/target/scala-2.10"
    BASE_DIR="$HOME/Private/studies/distributed-systems/wiki-embeddings/datasets"
fi

${SPARK_HOME}/bin/spark-submit \
   --master "local[*]" \
   --driver-memory 8g \
   --class pl.edu.pg.eti.Main \
${TARGET_DIR}/wiki-embeddings.jar \
   --baseDir ${BASE_DIR} | tee run.log