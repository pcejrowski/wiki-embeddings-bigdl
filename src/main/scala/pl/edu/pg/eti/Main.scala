package pl.edu.pg.eti

import java.io.File

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import pl.edu.pg.eti.CategoriesCalculator.Distance
import pl.edu.pg.eti.DataSets.CategoryName

import scala.reflect.ClassTag

object Main {

  def main(args: Array[String]): Unit = {
    ConfigParser
      .configParser
      .parse(args, TextClassificationParams())
      .foreach { param =>
        cf(param)
//        generateEmbeddings(param)
//        training(param)
//        testing(param)
//        generateTFIDF(param)
      }
  }

  private def generateTFIDF(implicit params: TextClassificationParams) = {
    val conf = new SparkConf()
      .setAppName("wiki-embeddings")


    val ss = SparkSession.builder()
      .appName("wiki-embeddings")
      .config("spark.task.maxFailures", "1")
      .getOrCreate()
    implicit val sc = ss.sparkContext
    val gen = new TFxIDFGenerator(ss)
    val catCalc = new CategoriesCalculator

    val dataSets = new DataSets()
    val artDictBC = ss.sparkContext.broadcast(dataSets.articlesDict())
    val corpus = ss
      .sparkContext
      .parallelize(dataSets.corpus().toSeq)
      .map { case (title, content) => artDictBC.value.getOrElse(title.replace(' ', '_'), -1) -> content }

    val tfIdfs = loadOrCalculate("tfidfs") {
      gen.generateTfIdfRepresentation(corpus)
    }

    val cats = loadOrCalculate("cats") {
      val catMembership = dataSets.articleCategories()
      catCalc.calcCategoriesRepresentations(tfIdfs, catMembership)
    }

    val distances = loadOrCalculate("distances") {
      val filter = dataSets.categoriesFilter()
      val data = cats.filter(x => filter.contains(x._1))
      catCalc.calcCatsDistance(data)
    }

    val distancesFinal: RDD[(CategoryName, CategoryName, Distance)] = loadOrCalculate("distances-final") {
      val catsDict = dataSets.categoriesDictRaw()
      val data = distances
        .map { case (cat1Id, cat2Id, dist) => (catsDict.getOrElse(cat1Id, ""), catsDict.getOrElse(cat2Id, ""), dist) }
        .collect()
        .sortBy(_._3)

      sc.parallelize(data, 1)
    }


  }

  def loadOrCalculate[T: ClassTag](name: String)(op: => RDD[T])(implicit params: TextClassificationParams, sc: SparkContext): RDD[T] = {
    val baseDir = new File(params.baseDir, "target")
    val objectFiles = new File(baseDir, name)
    if (objectFiles.exists()) {
      sc.objectFile[T](objectFiles.getAbsolutePath)
    } else {
      val result = op
      result.saveAsObjectFile(objectFiles.getAbsolutePath)
      result.saveAsTextFile(new File(baseDir, name + "-txt").getAbsolutePath)
      result
    }
  }

  def load[T: ClassTag](name: String)(implicit params: TextClassificationParams, sc: SparkContext): RDD[T] = {
    val baseDir = new File(params.baseDir, "target")
    val objectFiles = new File(baseDir, name)
    sc.objectFile[T](objectFiles.getAbsolutePath)
  }


  private def training(params: TextClassificationParams) = new TextClassifier(params).train()
  private def testing(params: TextClassificationParams) = new TextClassifier(params).test()
  private def generateEmbeddings(params: TextClassificationParams) = new TextClassifier(params).generateEmbeddings(params)
  private def cf(params: TextClassificationParams) = new CollaborativeFiltering(params).evaluate()
}