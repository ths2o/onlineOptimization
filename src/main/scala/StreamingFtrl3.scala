/**
  * Created by Taehee on 2018. 5. 22..
  */



import breeze.linalg.SparseVector
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.log4j.{Level, LogManager}
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010._

import scala.util.parsing.json.JSON.parseFull

object StreamingFtrl3 {

  /*

/home/ths2o717/spark-2.3.0-bin-hadoop2.7/bin/spark-submit \
--class StreamingFtrl3 \
--executor-memory 7G \
--driver-memory 7G \
/home/ths2o717/project/onlineOptimization/target/scala-2.11/followTheRegularizedLeader-assembly-0.1.0-SNAPSHOT.jar

~/Documents/project/spark-2.3.0-bin-hadoop2.7/bin/spark-submit \
--class StreamingFtrl3 \
~/Documents/project/onlineOptimization/target/scala-2.11/followTheRegularizedLeader-assembly-0.1.0-SNAPSHOT.jar


   */

  val sc = SparkContext.getOrCreate()
  val conf =sc.getConf.setMaster("local[2]").set("spark.driver.allowMultipleContexts", "true")
  val ssc = new StreamingContext(conf, Seconds(5))
  ssc//.checkpoint("/Users/Taehee/Documents/project/temp")

  val log = LogManager.getRootLogger
  log.setLevel(Level.WARN)

  //LogManager.getRootLogger().setLevel(Level.INFO)


  val kafkaParams = Map[String, Object](
    "bootstrap.servers" -> "localhost:9092,10.146.0.2:9092",
    "key.deserializer" -> classOf[StringDeserializer],
    "value.deserializer" -> classOf[StringDeserializer],
    "group.id" -> "use_a_separate_group_id_for_each_stream",
    "auto.offset.reset" -> "latest",
    "enable.auto.commit" -> (false: java.lang.Boolean)
  )

  val topics = Array("connect-test")
  val stream = KafkaUtils.createDirectStream[String, String](
    ssc,
    PreferConsistent,
    Subscribe[String, String](topics, kafkaParams)
  )

  // Create a local StreamingContext with two working thread and batch interval of 1 second.
  // The master requires 2 cores to prevent from a starvation scenario.




  val bb = stream.map(record => record.value).
    flatMap{x=> parseFull(x).get.asInstanceOf[Map[String, Any]].get("payload")}//.repartition(1)

  var i = 0
  val parse = bb.map{ x=>


    def mapToSparseVector(kv : Map[Int, Double], n:Int) = {
      val vec = SparseVector.zeros[Double](n)
      kv.keys.foreach(i => vec(i) = kv(i))
      vec
    }

    def libSvmParser(libSvm:String) = {
      val split = libSvm.split(" +")
      val label = split.take(1)(0) toInt
      val feature = split.drop(1).map{s=>
        val kv = s.split(":")
        val (k, v) = (kv(0), kv(1))
        (k.toInt -> v.toDouble)
      }.toMap
      (label, mapToSparseVector(feature, Int.MaxValue))
    }

    i +=1
    (i, libSvmParser(x.toString))
  }


  //var modelParam = new Ftrl().setAlpha(5).setBeta(1).setL1(1.5).setL2(0).save()

  val model = new FtrlSpark().
    setAlpha(10).
    setBeta(10).
    setL1(3).setL2(0)

  val ss = parse.foreachRDD{ x=>

    model.update(x.map(x=> x._2))

    val summary = model.bufferSummary(0.5)

    val summaryString = Array(
      "loss : " + "%.5f".format(summary._1),
      "precision : " + "%.5f".format(summary._2),
      "AUC : " + "%.5f".format(summary._3),
      "Non-zero Coef : " + summary._4,
      "Sample Count :" + model.i
    )

    println(summaryString.mkString(",  "))


    //println(model.nonZeroCoef)

    //log.warn(model.globalW)
    //println()
  }

  //parse.print()



/*
  val aa = parse.repartition(1).updateStateByKey[FtrlParam]{
    def updateFunction(newData:Seq[(Int, SparseVector[Double])], ftrlParam:Option[FtrlParam])= {

      var result: Option[FtrlParam] = ftrlParam
      //val ftrl = new Ftrl().setAlpha(5).setBeta(1).setL1(1.5).setL2(0)

      if(newData.isEmpty) {
        result = Some(ftrlParam.get)
      }
      else{
        newData.foreach { x => {
          if(ftrlParam.isEmpty){
            result = Some(new Ftrl().setAlpha(5).setBeta(1).setL1(1.5).setL2(0).update(x).save())
          }else{
            result = Some(
              new Ftrl().setAlpha(5).setBeta(1).setL1(0).setL2(0).
                load(ftrlParam.get).
                update(x).
                save()
            ) // update and return the value
          }
        } }
      }
      result
    }
    val func = updateFunction _
    func
  }


  val cc= aa.map(x=> (x._2.weight, x._2.i))
*/
  //parse.print()
  //ss.print()


  def main(args: Array[String]): Unit = {

    ssc.start()             // Start the computation
    ssc.awaitTermination()




  }
}
