package pl.edu.pg.eti

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import pl.edu.pg.eti.CategoriesCalculator.Distance
import pl.edu.pg.eti.DataSets.{ArticleId, CategoryId}

/**
  * Created by wpitula on 6/4/17.
  */
class CategoriesCalculator {

  def calcCategoriesRepresentations(articles: RDD[(ArticleId, Vector)],
                                    categoryMembership: Map[ArticleId, Set[CategoryId]],
                                    strategy: Iterable[Vector] => Vector = CategoriesCalculator.vectorAvg): RDD[(CategoryId, Vector)] = {


    val sc = articles.sparkContext
    val catMemBC = sc.broadcast(categoryMembership)

    articles
      .flatMap { case (id, vect) => catMemBC.value.getOrElse(id, Seq()).map(_ -> vect) }
      .groupByKey()
      .map { case (catId, vectors) => (catId, strategy(vectors)) }
  }

  def calcCatsDistance(data: RDD[(CategoryId, Vector)], strategy: (Vector, Vector) => Double = Vectors.sqdist): RDD[(CategoryId, CategoryId, Distance)] = {
    data
      .cartesian(data)
      .filter(x => x._1._1 < x._2._1) // so we can triangular matrix instead of full
      .map { case ((caId1, vec1), (catId2, vec2)) => (caId1, catId2, strategy(vec1, vec2)) }

  }


}

object CategoriesCalculator {

  type Distance = Double

  def vectorAvg(data: Iterable[Vector]): Vector = {
    val sum: BV[Distance] = data
      .map(x => BV(x.toArray))
      .reduce(_ + _)
    val avg: BV[Distance] = sum :/ data.size.toDouble
    Vectors.dense(avg.toArray)
  }
}


