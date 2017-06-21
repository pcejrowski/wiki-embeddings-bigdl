package pl.edu.pg.eti

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.{Logger, LoggerFactory}
import pl.edu.pg.eti.CategoriesCalculator.Distance
import pl.edu.pg.eti.DataSets.{CategoryId, CategoryName}

class CollaborativeFiltering(param: TextClassificationParams) {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def evaluate(): Unit = {
    val conf: SparkConf = new SparkConf()
      .setAppName("songs-collaborative-filtering")
      .setMaster("local[4]")
    implicit val sc: SparkContext = new SparkContext(conf)
    implicit val params = param

    val source: RDD[(CategoryName, CategoryName, Distance)] = Main.load[(CategoryName, CategoryName, Distance)]("distances-final")
    val dataSets = new DataSets()
    val dict: Predef.Map[CategoryId, CategoryName] = dataSets.categoriesDictRaw()
    val reversedDict: Predef.Map[CategoryName, CategoryId] = dict.map { case (cid, ncid) => ncid -> cid }

    val ratings: RDD[Rating] = source
      .map(r => Rating(
        reversedDict.getOrElse(r._1, 0),
        reversedDict.getOrElse(r._2, 0),
        r._3.toDouble))
    val filter: Set[CategoryId] = dataSets.categoriesFilter()

    ALS
      .train(ratings, 8, 6)
      .recommendUsersForProducts(5)
      .filter(x => filter.contains(x._1))
      .flatMap(x => x._2.map(r => s"${dict.getOrElse(r.product, "")},${dict.getOrElse(r.user, "")},${r.rating}"))
      .repartition(1)
      .saveAsTextFile("datasets/recommendations.txt")
  }
}
