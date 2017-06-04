package pl.edu.pg.eti

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

class DataSets(val baseDir: String = "datasets") {

  import DataSets._

  private val toStringOnInt: Array[String] => ((String, Int)) = tokens => tokens(0) -> tokens(1).toInt

  private val toIntOnIntSet: Array[String] => ((Int, Set[Int])) = { tokens =>
    val intTokens: Array[Int] = tokens.map(_.toInt)
    intTokens.head -> intTokens.tail.toSet
  }

  def loadData(): Array[(Content, NewCategoryIdF)] = {
    val dataSets = new DataSets(baseDir)
    val texts: Map[Title, Content] = dataSets.corpus()
    val articleLabels: Map[Title, ArticleId] = dataSets.articlesDict()
    val categoriesLabels: Map[CategoryId, NewCategoryId] = dataSets.categoriesDict()
    val articleCategories: Map[ArticleId, Set[CategoryId]] = dataSets.articleCategories()

    texts
      .toArray
      .flatMap { case (title, content) =>
        for {
          articleId <- articleLabels.get(title)
          articleCats <- articleCategories.get(articleId)
        } yield articleCats.map(catId => content -> categoriesLabels.getOrElse(catId, 0).toFloat)
      }
      .flatten
      .filter { case (content, id) => id != .0f }
  }


  def articlesDict(fileName: String = "articles_dict"): Map[Title, ArticleId] = {
    load(fileName, toStringOnInt)
      .map {
        case (title, id) => title.replace(' ', '_') -> id
      }
      .toMap
  }

  def categoriesDict(fileName: String = "cats_dict"): Map[CategoryId, NewCategoryId] = {
    load(fileName, toStringOnInt)
      .zipWithIndex
      .map {
        case ((categoryName, categoryId), newCategoryId) => categoryId -> (newCategoryId + 1)
      }
      .toMap
  }

  def categoriesDictRaw(fileName: String = "cats_dict"): Map[CategoryId, CategoryName] = {
    load(fileName, toStringOnInt).map(x => (x._2, x._1)).toMap
  }

  def articleCategories(fileName: String = "categories"): Map[ArticleId, Set[CategoryId]] = {
    load(fileName, toIntOnIntSet).toMap
  }
  def categoriesFilter(fileName: String = "cats_filter"): Set[CategoryId] = {
    load(fileName, _.head.toInt).toSet
  }


  def corpus(fileName: String = "corpus.txt"): Map[Title, Content] = {
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

  private def load[T](fileName: String, transformer: Array[String] => T): List[T] = {
    Source.fromFile(s"$baseDir/$fileName")
      .getLines()
      .map(_.split("\\s+"))
      .map(transformer)
      .toList
  }
}

object DataSets {
  type Title = String
  type Content = String
  type ArticleId = Int
  type CategoryId = Int
  type NewCategoryId = Int
  type NewCategoryIdF = Float
  type CategoryName = String
}