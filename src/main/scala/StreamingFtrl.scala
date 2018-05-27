/**
  * Created by Taehee on 2018. 5. 22..
  */



import breeze.linalg.SparseVector
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.streaming.dstream.DStream



object StreamingFtrl {



  // Create a local StreamingContext with two working thread and batch interval of 1 second.
  // The master requires 2 cores to prevent from a starvation scenario.

  val sc = SparkContext.getOrCreate()
  val conf =sc.getConf.set("spark.driver.allowMultipleContexts", "true")
  val ssc = new StreamingContext(conf, Seconds(1))
  ssc.checkpoint("/Users/Taehee/Documents/project/temp")


  val lines = ssc.socketTextStream("localhost", 8888)
  val parse = lines.map{ x=>

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

    (1, libSvmParser(x))
  }


  val aa = parse.updateStateByKey[FtrlParam]{
    def updateFunction(newData:Seq[(Int, SparseVector[Double])], ftrlParam:Option[FtrlParam])= {

      var result: Option[FtrlParam] = null
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

    updateFunction _
  }


  val bb= aa.map(x=> (x._2.weight, x._2.i))
  bb.print()


  def main(args: Array[String]): Unit = {

    ssc.start()             // Start the computation
    ssc.awaitTermination()




  }
}
