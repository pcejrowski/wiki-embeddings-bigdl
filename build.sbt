name := "wiki-embeddings"

version := "1.0"

scalaVersion := "2.10.6"

val sparkVersion = "2.1.0"

mainClass in assembly := Some("pl.edu.pg.eti.Main")
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
resolvers += Resolver.mavenLocal

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.0" % "0.2.0-SNAPSHOT"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"

assemblyJarName in assembly := s"${name.value}.jar"
