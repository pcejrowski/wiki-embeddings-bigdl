package pl.edu.pg.eti

import com.intel.analytics.bigdl.example.utils.WordMeta
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import pl.edu.pg.eti.TextAnalyzer.Word
import pl.edu.pg.eti.TextClassifier.WordVec

import scala.collection.mutable.{Map => MMap}
import scala.io.Source
import scala.language.existentials

class Word2Vec(val gloveDir: String) {

  import Word2Vec._

  def loadWord2Vec(word2Meta: Map[Word, WordMeta]): Map[WordMetaIdxF, WordVec] = {
    val preWord2Vec = MMap[WordMetaIdxF, WordVec]()
    val filename = s"$gloveDir/simple-wiki-embeddings.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.drop(1).map(_.toFloat)
        preWord2Vec.put(word2Meta(word).index.toFloat, coefs)
      }
    }
    preWord2Vec.toMap
  }
}

object Word2Vec {
  type WordMetaIdxF = Float
}
