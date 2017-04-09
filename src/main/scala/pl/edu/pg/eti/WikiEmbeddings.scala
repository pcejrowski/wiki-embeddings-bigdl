package pl.edu.pg.eti

object WikiEmbeddings {

  def main(args: Array[String]): Unit = {
    ConfigParser
      .configParser
      .parse(args, TextClassificationParams())
      .foreach { param =>
        new TextClassifier(param).train()
      }
  }
}