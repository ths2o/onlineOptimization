/**
  * Created by Taehee on 2018. 5. 22..
  */



import breeze.linalg.SparseVector
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.streaming.dstream.DStream


object StreamingFtrl {


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
/*
  def updateFunction(data:Seq[(Int, SparseVector[Double])], ftrl:Option[Ftrl])= {
    val model = ftrl.get
    val update = data.foreach(x=> model.update(x))
    Some(model)
  }
*/
  // Create a local StreamingContext with two working thread and batch interval of 1 second.
  // The master requires 2 cores to prevent from a starvation scenario.

  val sc = SparkContext.getOrCreate()
  val conf =sc.getConf.set("spark.driver.allowMultipleContexts", "true")
  val ssc = new StreamingContext(conf, Seconds(1))


  val ftrl = new Ftrl().setAlpha(5).setBeta(1).setL1(1.5).setL2(0)



  val lines = ssc.socketTextStream("localhost", 9999)
  val parse = lines.map{ x=>
    libSvmParser(x)._1
  }

  def updateFunction(data:Seq[Int], ftrl:Option[Ftrl])= {
    val model = ftrl.get
    val update = 1
    Some(model)
  }

  //val aa = parse.updateStateByKey[Ftrl](updateFunction)

  val weight = parse.map{x=>
    ftrl.update(x)
    ftrl.weight.toString() + "\n" + ftrl.i.toString
  }

  weight.print()


  def main(args: Array[String]): Unit = {

    ssc.start()             // Start the computation
    ssc.awaitTermination()




  }
}
