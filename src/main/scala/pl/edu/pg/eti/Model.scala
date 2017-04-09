package pl.edu.pg.eti

import com.intel.analytics.bigdl.nn._

class Model(param: TextClassificationParams) {
  def buildModel(classNum: Int): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(param.embeddingDim, 1, param.maxSequenceLength)))

      .add(SpatialConvolution(param.embeddingDim, 128, 5, 1))
      .add(ReLU())

      .add(SpatialMaxPooling(5, 1, 5, 1))

      .add(SpatialConvolution(128, 128, 5, 1))
      .add(ReLU())

      .add(SpatialMaxPooling(5, 1, 5, 1))

      .add(SpatialConvolution(128, 128, 5, 1))
      .add(ReLU())

      .add(SpatialMaxPooling(35, 1, 35, 1))

      .add(Reshape(Array(128)))
      .add(Linear(128, 1000))
      .add(Linear(1000, classNum))
      .add(LogSoftMax())

  }

}
