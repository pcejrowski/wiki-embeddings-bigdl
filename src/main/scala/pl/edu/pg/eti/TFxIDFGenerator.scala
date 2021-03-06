package pl.edu.pg.eti

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import pl.edu.pg.eti.DataSets.{ArticleId, Content, Title}
import pl.edu.pg.eti.TextAnalyzer.Word

/**
  * Created by wpitula on 5/1/17.
  */
class TFxIDFGenerator(ss: SparkSession) {

  import TFxIDFGenerator._

  def generateTfIdfRepresentation(corpus: RDD[(ArticleId, Content)]): RDD[(ArticleId, ArticleTFIDF)] = {

    val df = ss.createDataFrame(corpus).toDF("id", "content")
    val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("words")
    val wordsData = tokenizer.transform(df)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.rdd.map(r => (r.getAs[Int]("id"), r.getAs[Vector]("features")))
  }

}

object TFxIDFGenerator {
  type Article = Iterable[Word]
  type ArticleTFIDF = Vector
}