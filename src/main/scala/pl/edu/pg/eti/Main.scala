package pl.edu.pg.eti

import java.io.File

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Main {

  def main(args: Array[String]): Unit = {
    ConfigParser
      .configParser
      .parse(args, TextClassificationParams())
      .foreach { param =>
        generateTFIDF(param)
      }
  }


  private def classifyText(params: TextClassificationParams) = new TextClassifier(params).train()

  private def generateTFIDF(params: TextClassificationParams) = {
    val conf = new SparkConf()
      .setAppName("wiki-embeddings")


    val ss = SparkSession.builder()
      .appName("wiki-embeddings")
      .config("spark.task.maxFailures", "1")
      .getOrCreate()
    val gen = new TFxIDFGenerator(ss)
    val dataSets = new DataSets()
    val corpus = ss.sparkContext.parallelize(dataSets.corpus().toSeq)

    val tfidfs = gen.generateTfIdfRepresentation(corpus)

    tfidfs.saveAsTextFile(new File(params.baseDir, "tfidfs").getAbsolutePath)
  }


}