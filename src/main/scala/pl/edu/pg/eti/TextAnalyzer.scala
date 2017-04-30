package pl.edu.pg.eti

import com.intel.analytics.bigdl.example.utils.{SimpleTokenizer, WordMeta}
import org.apache.spark.rdd.RDD
import pl.edu.pg.eti.DataSets.{Content, NewCategoryIdF}
import pl.edu.pg.eti.Word2Vec.WordMetaIdxF

class TextAnalyzer(val gloveDir: String, param: TextClassificationParams) {

  import TextAnalyzer._

  def analyzeTexts(dataRdd: RDD[(Content, NewCategoryIdF)]): (Map[Word, WordMeta], Map[WordMetaIdxF, Array[Float]]) = {
    val frequencies = dataRdd
      .flatMap { case (text: String, label: Float) => SimpleTokenizer.toTokens(text) }
      .map(word => (word, 1))
      .reduceByKey(_ + _)
      .sortBy(-_._2)
      .collect()
      .slice(10, param.maxWordsNum) // ???

    val indexes = Range(1, frequencies.length)
    val word2Meta = frequencies
      .zip(indexes)
      .map { item => (item._1._1, WordMeta(item._1._2, item._2)) }
      .toMap
    (word2Meta, new Word2Vec(gloveDir).loadWord2Vec(word2Meta))
  }

}

object TextAnalyzer {
  type Word = String
}