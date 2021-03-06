package pl.edu.pg.eti

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.example.utils.SimpleTokenizer
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, File, T}
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}
import pl.edu.pg.eti.CategoriesCalculator.Distance
import pl.edu.pg.eti.DataSets._
import pl.edu.pg.eti.Word2Vec.WordMetaIdxF

import scala.collection.mutable.{Map => MMap}
import scala.language.existentials

class TextClassifier(param: TextClassificationParams) {

  import TextClassifier._

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val gloveDir = s"${param.baseDir}/glove.6B/"
  val textDataDir = s"${param.baseDir}"
  var classNum: Int = new DataSets(param.baseDir).categoriesDict().size

  def generateEmbeddings(implicit param: TextClassificationParams): Unit = {
    val conf = Engine
      .createSparkConf()
      .setAppName("wiki-embeddings")
      .set("spark.task.maxFailures", "1")

    implicit val sc = new SparkContext(conf)

    val rdd: RDD[(Array[WordVec], NewCategoryIdF)] = getVectorizedRdd(sc)

    val catCalc = new CategoriesCalculator
    val datasets = new DataSets()
    val dict: Map[CategoryId, NewCategoryId] = datasets.categoriesDict()
    val reversedDict: Map[NewCategoryId, CategoryId] = dict.map { case (cid, ncid) => ncid -> cid }
    val filter: Set[NewCategoryId] = datasets.categoriesFilter().map(dict.getOrElse(_, 0))
    val catsDict: Map[CategoryId, CategoryName] = datasets.categoriesDictRaw()

    val categoryCentroids: RDD[(CategoryId, linalg.Vector)] = Main.loadOrCalculate("category-centroids") {
      rdd
        .filter { case (_, label) => filter.contains(label.toInt) }
        .map { case (vectors, label) =>
          val vector = vectors
            .reduce((a1, a2) => a1
              .zip(a2)
              .map { case (v1, v2) => v1 + v2 }
            )
          val dense: linalg.Vector = Vectors.dense(vector.map(_.toDouble))
          label.toInt -> dense
        }
        .groupByKey()
        .map { case (ncid, vectors: Iterable[linalg.Vector]) =>
          val centroid: linalg.Vector = vectors
            .reduce { (v1: linalg.Vector, v2: linalg.Vector) =>
              Vectors
                .dense(
                  v1.toArray
                    .zip(v2.toArray)
                    .map { case (a1, a2) => a1 + a2 })
            }
          Vectors.dense(centroid.toArray.map(x => x / vectors.size))
          (reversedDict.getOrElse(ncid, 0), centroid)
        }
        .repartition(1)
    }
    val categoryCentroidsWithNames = Main.loadOrCalculate("category-centroids-with-names") {
      categoryCentroids
        .map { case (catId, vector) => (catsDict.getOrElse(catId, ""), vector) }
        .repartition(1)
    }
    val distances: RDD[(CategoryId, CategoryId, Distance)] = catCalc.calcCatsDistance(categoryCentroids)
    val finalDistances: RDD[(CategoryName, CategoryName, Distance)] = Main.loadOrCalculate("distances-final-embeddings") {
      val data = distances
        .map { case (cat1Id, cat2Id, dist) =>
          (catsDict.getOrElse(cat1Id, ""),
            catsDict.getOrElse(cat2Id, ""),
            dist)
        }
        .collect()
        .sortBy(_._3)

      sc.parallelize(data, 1)
    }
  }

  def test(): Unit = {
    val conf = Engine
      .createSparkConf()
      .setAppName("wiki-embeddings")
      .set("spark.task.maxFailures", "1")

    val sc = new SparkContext(conf)
    val sampleRDD: RDD[Sample[WordMetaIdxF]] = getSampleRdd(sc)
    val testRDD: RDD[Sample[WordMetaIdxF]] = sampleRDD.sample(withReplacement = false, 0.2)

    Engine.init
    val loaded: Module[WordMetaIdxF] = File.load[Module[Float]]("model.serialized")
    val x: RDD[(CategoryId, Sample[WordMetaIdxF])] = loaded
      .predictClass(testRDD)
      .zip(testRDD)
    val y: Array[(ValidationResult, ValidationMethod[WordMetaIdxF])] = loaded.evaluate(testRDD, Array(new Top5Accuracy[WordMetaIdxF]))
    y.foreach { case (result, method) => println(result.result()) }
    sc.stop()
  }

  def train(): Unit = {
    val conf = Engine
      .createSparkConf()
      .setAppName("wiki-embeddings")
      .set("spark.task.maxFailures", "1")

    val trainingSplit: Double = param.trainingSplit

    val sc = new SparkContext(conf)
    Engine.init
    val sampleRDD: RDD[Sample[WordMetaIdxF]] = getSampleRdd(sc)
    val Array(trainingRDD, validationRDD, _) = sampleRDD.randomSplit(Array(trainingSplit, (1 - trainingSplit) / 2, (1 - trainingSplit) / 2))

    val optimizer = Optimizer(
      model = new Model(param).buildModel(classNum),
      sampleRDD = trainingRDD,
      criterion = new ClassNLLCriterion[WordMetaIdxF](),
      batchSize = param.batchSize
    )

    val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)

    val model: Module[WordMetaIdxF] = optimizer
      .setState(state)
      .setOptimMethod(new Adagrad())
      .setDropMoudleProperty(0.1, 0.2)
      .setValidation(Trigger.everyEpoch, validationRDD, Array(new Top5Accuracy[WordMetaIdxF]), param.batchSize)
      .setEndWhen(Trigger.maxEpoch(10))
      .optimize()


    val modelFile = "model.serialized"
    model.save(modelFile, overWrite = true)

    sc.stop()
  }

  def getSampleRdd(sc: SparkContext): RDD[Sample[WordMetaIdxF]] = {
    val sequenceLen: Int = param.maxSequenceLength
    val embeddingDim: Int = param.embeddingDim

    val sampleRDD: RDD[Sample[WordMetaIdxF]] = getVectorizedRdd(sc)
      .map { case (input: Array[Array[WordMetaIdxF]], label: WordMetaIdxF) =>
        Sample(featureTensor =
          Tensor(input.flatten, Array(sequenceLen, embeddingDim)).transpose(1, 2).contiguous(),
          labelTensor = Tensor(Array(label), Array(1))
        )
      }
    sampleRDD
  }


  def getVectorizedRdd(sc: SparkContext): RDD[(Array[WordVec], NewCategoryIdF)] = {
    val sequenceLen: Int = param.maxSequenceLength
    val embeddingDim: Int = param.embeddingDim

    val data: Array[(Content, NewCategoryIdF)] = new DataSets(textDataDir).loadData()
    val dataRdd = sc.parallelize(data, param.partitionNum)
    val (word2Meta, word2Vec) = new TextAnalyzer(gloveDir, param).analyzeTexts(dataRdd)
    val word2MetaBC = sc.broadcast(word2Meta)
    val word2VecBC = sc.broadcast(word2Vec)

    import SimpleTokenizer._
    val vectorizedRdd: RDD[(Array[WordVec], NewCategoryIdF)] = dataRdd
      .map { case (text: Content, label: NewCategoryIdF) =>
        (toTokens(text, word2MetaBC.value), label)
      }
      .map { case (tokens: Array[WordMetaIdxF], label: NewCategoryIdF) =>
        (shaping(tokens, sequenceLen), label)
      }
      .map { case (tokens: Array[WordMetaIdxF], label: NewCategoryIdF) =>
        (vectorization(tokens, embeddingDim, word2VecBC.value), label)
      }
    vectorizedRdd
  }
}

object TextClassifier {
  type WordVec = Array[Float]
}


/**
  * 1) Ograniczony zbiór kategorii świadomie dobranej (~10 kat)
  * //  * 2) na kategoriach zrobić reprezentacje na słowach (TF-IDF)
  * //  * 3) porównać z wynikami uzyskanymi poprzez reprezentacje na Embeddingsach
  * 4) wrzucenie reprezentacji na NN
  */
