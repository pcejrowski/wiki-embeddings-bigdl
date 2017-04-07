name := "wiki-embeddings"

version := "1.0"

scalaVersion := "2.10.5"

val sparkVersion = "1.5.1"

mainClass in Compile := Some("pl.edu.pg.eti.WikiEmbeddings")
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl" % "0.1.0-SNAPSHOT"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
