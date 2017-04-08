package pl.edu.pg.eti

import com.intel.analytics.bigdl.example.textclassification.WordMeta
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}

import scala.collection.mutable.{Map => MMap}
import scala.io.Source
import scala.language.existentials

class Word2Vec(val gloveDir: String) {
  def loadWord2Vec(word2Meta: Map[String, WordMeta]): Map[Float, Array[Float]] = {
    val preWord2Vec = MMap[Float, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.100d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.slice(1, values.length).map(_.toFloat)
        preWord2Vec.put(word2Meta(word).index.toFloat, coefs)
      }
    }
    preWord2Vec.toMap
  }
}
