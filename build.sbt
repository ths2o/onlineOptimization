organization := "com.sth"

name := "followTheRegularizedLeader"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.11.8"

//seq(webSettings :_*)

libraryDependencies ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2",

  "com.typesafe.akka" %% "akka-actor" % "2.5.12",
  "com.typesafe.akka" %% "akka-testkit" % "2.5.12" % Test,
  "com.typesafe.akka" %% "akka-stream" % "2.5.12",
  "com.typesafe.akka" %% "akka-stream-testkit" % "2.5.12" % Test,

  "org.apache.spark" %% "spark-core" % "2.3.0" % "provided",
  "org.apache.spark" %% "spark-streaming" % "2.3.0" % "provided",
  "org.apache.spark" %% "spark-streaming-kafka-0-10" % "2.3.0",

  "org.apache.spark" %% "spark-mllib" % "2.3.0" % "provided",
  "com.github.alexandrnikitin" %% "bloom-filter" % "latest.release"

)
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
resolvers += "Sonatype OSS Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots/"