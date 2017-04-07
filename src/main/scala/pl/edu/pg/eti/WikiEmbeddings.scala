package pl.edu.pg.eti

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object WikiEmbeddings {

  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf()
      .setAppName("wiki-embeddings")
      .setMaster("local[4]")
    val sc: SparkContext = new SparkContext(conf)

  }
}