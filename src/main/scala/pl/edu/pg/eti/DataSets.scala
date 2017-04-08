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
    val categoriesLabels: Map[Int, Int] = dataSets.categoriesDict()
    val articleCategories: Map[Int, Set[Int]] = dataSets.articleCategories()

    texts
      .flatMap { case (title, content) =>
        for {
          articleId <- articleLabels.get(title)
          articleCats <- articleCategories.get(articleId)
        } yield articleCats.map(catId => content -> categoriesLabels.getOrElse(catId, 0).toFloat)
      }
      .flatten
      .filter { case (content, id) => id != .0f }
      .toArray
  }


  def articlesDict(fileName: String = "articles_dict"): Map[String, Int] = {
    load(fileName, toStringOnInt)
      .map {
        case (title, id) => title.replace(' ', '_') -> id
      }
      .toMap
  }

  def categoriesDict(fileName: String = "cats_dict"): Map[Int, Int] = {
    load(fileName, toStringOnInt)
      .zipWithIndex
      .map {
        case ((categoryName, categoryId), newCategoryId) => categoryId -> (newCategoryId + 1)
      }
      .toMap
  }

  def articleCategories(fileName: String = "categories"): Map[Int, Set[Int]] = {
    load(fileName, toIntOnIntSet).toMap
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

  private def load[T, U](fileName: String, transformer: Array[String] => ((T, U))): List[(T, U)] = {
    Source.fromFile(s"$baseDir/$fileName")
      .getLines()
      .map(_.split("\\s+"))
      .map(transformer)
      .toList
  }
}