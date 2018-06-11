import java.io.{FileOutputStream, ObjectOutputStream}

import org.apache.hadoop.util.bloom.CountingBloomFilter
import org.apache.hadoop.util.hash.Hash.MURMUR_HASH
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

import scala.util.hashing.MurmurHash3

/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuTrain3 {





  /**
    *
    * id, click, hour, C1, banner_pos,
    * site_id, site_domain, site_category, app_id, app_domain,
    * app_category, device_id, device_ip, device_model, device_type,
    * device_conn_type, C14, C15, C16, C17,
    * C18, C19, C20, C21

    */

  /*

    ~/Documents/project/spark-2.3.0-bin-hadoop2.7/bin/spark-submit \
    --class AvazuTrain ~/Documents/project/onlineOptimization/target/scala-2.11/followTheRegularizedLeader-assembly-0.1.0-SNAPSHOT.jar

   */


  def hash(c: String) = {
    math.abs(MurmurHash3.stringHash(c, 1))
  }


  def main(args: Array[String]): Unit = {


    val sc = SparkContext.getOrCreate()
    LogManager.getRootLogger().setLevel(Level.OFF)

    val spark = SparkSession.builder().getOrCreate()



    import spark.implicits._



    val model = new FtrlSpark().setAlpha(5).setBeta(1).setL1(4).setL2(0)

    val aa = new CountingBloomFilter(100000, 7, MURMUR_HASH)

    val ind = spark.read.parquet("/Users/Taehee/Documents/project/avazu3").
      select("hour").distinct.map(x=> x.getAs[Int]("hour")).
      collect.sortBy(x=> x).drop(24)

    ind.foreach{x=>

      val train = spark.read.parquet(s"/Users/Taehee/Documents/project/avazu3/hour=$x").
        drop("id", "h", "d", "c14", "c17").
        withColumn("c14Ctr", expr("case when c14Ctr is null then 0 else c14Ctr end")).
        withColumn("c17Ctr", expr("case when c17Ctr is null then 0 else c17Ctr end")).
        withColumn("pubIdCtr", expr("case when pubIdCtr is null then 0 else pubIdCtr end")).
        withColumn("pubCategoryCtr", expr("case when pubCategoryCtr is null then 0 else pubCategoryCtr end")).
        sample(0.3)

      val t2 = train.map{
        case Row(
        pubCategory, pubId, click, c1, banner, deviceT, deviceCon,
        c15, c16, c18, c19, c20, c21, userId, pubDomain,
        c14Ctr, c17Ctr, pubIdCtr, pubCategoryCtr
        )=>

          val y = click.asInstanceOf[String]
          //val t = topic.asInstanceOf[DenseVector]

          //val tt = t.toArray.zipWithIndex.maxBy(x=> x._1)._2

          //val topicFeat = (0 to 9).map(x=> (hash("t" + x)%100, t(x))).toMap

          val contFeat = Array(
            ("c14Ctr", c14Ctr.asInstanceOf[Double]),
            ("c17Ctr", c17Ctr.asInstanceOf[Double]),
            ("pubIdCtr", pubIdCtr.asInstanceOf[Double]),
            ("pubCategoryCtr", pubCategoryCtr.asInstanceOf[Double])

          ).map(x=> (hash(x._1) % 100, x._2)).toMap

          val userFeat = Array(

            "userId" + userId.asInstanceOf[String]
          ).map(x=> (hash(x) % 100000 + 100, 1D)).toMap


          val catFeat = Array(
            //"1" + d.hour,
            //"d-" + day,
            //"h-" + hour,
            "c1-" + c1.asInstanceOf[String],
            "bp-" + banner.asInstanceOf[String],
            "pId-" + pubId.asInstanceOf[String],
            //"pd-" + pubDomain.asInstanceOf[String],
            "pc-" + pubCategory.asInstanceOf[String],
            "dt-" + deviceT.asInstanceOf[String],
            "dc-" + deviceCon.asInstanceOf[String],
            //"c14-" + c14.asInstanceOf[String],
            "c15-" + c15.asInstanceOf[String],
            "c16-" + c16.asInstanceOf[String],
            //"c17-" + c17.asInstanceOf[String],
            "c18-" + c18.asInstanceOf[String],
            "c19-" + c19.asInstanceOf[String],
            "c20-" + c20.asInstanceOf[String],
            "c21-" + c21.asInstanceOf[String]
            //"topic-" + tt
          ).map(x=> (hash(x) % 100000 + 110000, 1D)).toMap


          val interaction = contFeat.map{x=>
            Array(
              //"1" + d.hour,
              //"d-" + day,
              //"h-" + hour,
              "t" + x._1 + x + "pId-" + pubId.asInstanceOf[String],
              //"t" + x._1 + "pd-" + pubDomain.asInstanceOf[String],
              "t" + x._1 + x + "pc-" + pubCategory.asInstanceOf[String],
              "t" + x._1 + x + "c15-" + c15.asInstanceOf[String],
              "t" + x._1 + x + "c16-" + c16.asInstanceOf[String],
              "t" + x._1 + x + "c18-" + c18.asInstanceOf[String],
              "t" + x._1 + x + "c19-" + c19.asInstanceOf[String],
              "t" + x._1 + x + "c20-" + c20.asInstanceOf[String],
              "t" + x._1 + x + "c21-" + c21.asInstanceOf[String]
            ).map(k=> (hash(k) % 100000 + 210000, x._2))
          }.flatten.toMap



          val feat = contFeat ++ userFeat ++ catFeat ++ interaction

          (y.toInt, feat)

      }.rdd
      //println(x.mkString(" "))
      /*

      .map { x =>
        val k = new Key(x.getBytes())
        aa.add(k)
        val c = aa.approximateCount(k)
        (x, if (c >= 14) true else false)
      }.filter(x=> x._2).map(x=> x._1)

  */

      val t3 = t2.map(x=> (x._1, FtrlRun.mapToSparseVector(x._2, Int.MaxValue)))
      model.update(t3)

      val summary = model.bufferSummary(0.5)
      val summaryString = Array(
        "loss : " + "%.5f".format(summary._1),
        "precision : " + "%.5f".format(summary._2),
        "AUC : " + "%.5f".format(summary._3),
        "Non-zero Coef : " + summary._4,
        "Sample Count :" + x
      )

      println(summaryString.mkString(",  "))

      val oos = new ObjectOutputStream(new FileOutputStream("/Users/Taehee/Downloads/ftrlParam"))
      oos.writeObject(model.weight)
      oos.close

    }




  }

}
