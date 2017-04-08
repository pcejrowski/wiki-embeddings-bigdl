package pl.edu.pg.eti

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

class DataSets(val baseDir: String = "datasets") {
  private val toStringOnInt: Array[String] => ((String, Int)) = tokens => tokens(0) -> tokens(1).toInt

  private val toIntOnIntSet: Array[String] => ((Int, Set[Int])) = { tokens =>
    val intTokens: Array[Int] = tokens.map(_.toInt)
    intTokens.head -> intTokens.tail.toSet
  }

  def loadData(): Array[(String, Float)] = {
    val dataSets = new DataSets(baseDir)
    val texts: Map[String, String] = dataSets.corpus()
    val articleLabels: Map[String, Int] = dataSets.articlesDict()
    val articleCategories: Map[Int, Set[Int]] = dataSets.articleCategories()

    texts
      .flatMap { case (title, content) =>
        val articleId: Int = articleLabels.getOrElse(title, 0)
        val articleCategoryIds: Set[Int] = articleCategories.getOrElse(articleId, Set[Int]())
        articleCategoryIds
          .map(cat => content -> cat.toFloat)
      }
      .toArray
  }


  def articlesDict(fileName: String = "articles_dict"): Map[String, Int] = {
    load(fileName, toStringOnInt)
  }

  def categoriesDict(fileName: String = "cats_dict"): Map[String, Int] = {
    load(fileName, toStringOnInt)
  }

  def articleCategories(fileName: String = "categories"): Map[Int, Set[Int]] = {
    load(fileName, toIntOnIntSet)
  }

  def corpus(fileName: String = "corpus.txt"): Map[String, String] = {
    val fileLines: Iterator[String] = Source.fromFile(s"$baseDir/$fileName")
      .getLines()

    var result: mutable.Map[String, String] = mutable.Map[String, String]()
    var articleData: ListBuffer[String] = ListBuffer[String]()
    while (fileLines.hasNext) {
      val line: String = fileLines.next()
      if (line != "") {
        articleData += line
      } else {
        val title = articleData.head
        val content = articleData.tail.mkString(" ")
        articleData = ListBuffer[String]()
        result += (title -> content)
      }
    }
    result.toMap
  }

  private def load[T, U](fileName: String, transformer: Array[String] => ((T, U))): Map[T, U] = {
    Source.fromFile(s"$baseDir/$fileName")
      .getLines()
      .map(_.split("\t"))
      .map(transformer)
      .toMap
  }
}