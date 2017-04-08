package pl.edu.pg.eti

import scopt.OptionParser

/**
  * @param baseDir           The root directory which containing the training and embedding data
  * @param maxSequenceLength number of the tokens
  * @param maxWordsNum       maximum word to be included
  * @param trainingSplit     percentage of the training data
  * @param batchSize         size of the mini-batch
  * @param embeddingDim      size of the embedding vector
  */
case class TextClassificationParams(baseDir: String = "./datasets",
                                    maxSequenceLength: Int = 1000,
                                    maxWordsNum: Int = 20000,
                                    trainingSplit: Double = 0.8,
                                    batchSize: Int = 128,
                                    embeddingDim: Int = 100,
                                    partitionNum: Int = 4)
object ConfigParser {
  val configParser = new OptionParser[TextClassificationParams]("BigDL Example") {
    opt[String]('b', "baseDir")
      .required()
      .text("Base dir containing the training and word2Vec data")
      .action((x, c) => c.copy(baseDir = x))
    opt[String]('p', "partitionNum")
      .text("you may want to tune the partitionNum if run into spark mode")
      .action((x, c) => c.copy(partitionNum = x.toInt))
    opt[String]('s', "maxSequenceLength")
      .text("maxSequenceLength")
      .action((x, c) => c.copy(maxSequenceLength = x.toInt))
    opt[String]('w', "maxWordsNum")
      .text("maxWordsNum")
      .action((x, c) => c.copy(maxWordsNum = x.toInt))
    opt[String]('l', "trainingSplit")
      .text("trainingSplit")
      .action((x, c) => c.copy(trainingSplit = x.toDouble))
    opt[String]('z', "batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x.toInt))
  }

}
