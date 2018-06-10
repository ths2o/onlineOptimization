import java.io.{FileOutputStream, ObjectOutputStream}

import org.apache.hadoop.util.bloom.CountingBloomFilter
import org.apache.hadoop.util.bloom.Key
import org.apache.hadoop.util.hash.Hash.MURMUR_HASH
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import scala.util.hashing.MurmurHash3

/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuTrain5 {





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

    ~/spark-2.3.0-bin-hadoop2.7/bin/spark-submit \
    --class AvazuTrain5 ~/project/onlineOptimization/target/scala-2.11/followTheRegularizedLeader-assembly-0.1.0-SNAPSHOT.jar

   */


  def hash(c: String) = {
    math.abs(MurmurHash3.stringHash(c, 1))
  }


  def main(args: Array[String]): Unit = {


    val sc = SparkContext.getOrCreate()
    LogManager.getRootLogger().setLevel(Level.OFF)

    val spark = SparkSession.builder().getOrCreate()



    import spark.implicits._



    val model = new FtrlSpark().setAlpha(5).setBeta(1).setL1(1.5).setL2(0)

    //val ind = spark.read.parquet("/Users/Taehee/Documents/project/avazu5").
    val ind = spark.read.parquet("./avazu/trainTest").
      select("hour").distinct.map(x=> x.getAs[Int]("hour")).
      collect.sortBy(x=> x).drop(24).dropRight(24)


    ind.foreach{x=>

      //val train = spark.read.parquet(s"/Users/Taehee/Documents/project/avazu5/hour=$x").
      val train = spark.read.parquet(s"./avazu/trainTest/hour=$x").
        withColumn("hourCount", expr("case when hourCount is null then 0 else hourCount end")).
        withColumn("dayCount", expr("case when dayCount is null then 0 else dayCount end")).
        withColumn("h", expr("substring(dateTime, 12, 2)")).
        sample(0.3)

      val t2 = train.mapPartitions{ part =>

        val filter = new CountingBloomFilter(100000, 7, MURMUR_HASH)

        part.map{
          case Row(
          userId, dt, dateTime, id, click, c1,pubId, pubDomain, pubCategory, impression,
          pub, hourCount, dayCount, h
          )=>

            val y = click.asInstanceOf[String]
            //val t = topic.asInstanceOf[DenseVector]

            //val tt = t.toArray.zipWithIndex.maxBy(x=> x._1)._2

            //val topicFeat = (0 to 9).map(x=> (hash("t" + x)%100, t(x))).toMap

            val a1 = if(dayCount.asInstanceOf[Long] >= 500) userId.toString else dayCount.asInstanceOf[Long].toString
            val a2 = if(hourCount.asInstanceOf[Long] >= 30) userId.toString else hourCount.asInstanceOf[Long].toString

            val userFeat = Array(

              ("userIdCtr" + a1),
              ("userIdHctr"+ a2)
            ).map(x=> (hash(x) % 1000000, 1D)).toMap


            val pubFeat = Array(
              ("pub" + pub.asInstanceOf[String])
            ).map(x=> (hash(x) % 1000000 + 1000000, 1D)).toMap

            val impFeat = Array(
              ("imp" + impression.asInstanceOf[String])
            ).map(x=> (hash(x) % 1000000 + 2000000, 1D)).toMap

            val hourFeat = Array(
              ("hour" + h.asInstanceOf[String])
            ).map(x=> (hash(x) % 1000000 + 3000000, 1D)).toMap


            val interaction1 = userFeat.map{u=>
              u._1.toString + h.toString
            }.map(x=> (hash(x) % 1000000 + 4000000, 1D)).toMap


            val interaction2 = pubFeat.map{p=>
              p.toString + h.toString
            }.map(x=> (hash(x) % 1000000 + 5000000, 1D)).toMap

            /*
            val interaction1 = userFeat.map{u=>
              pubFeat.map{p=>
                u._1.toString + p._1.toString
              }
            }.flatten.map(x=> (hash(x) % 1000000 + 3000000, 1D)).toMap

            val interaction2 = userFeat.map{u=>
              impFeat.map{i=>
                u._1.toString + i._1.toString
              }
            }.flatten.map(x=> (hash(x) % 1000000 + 4000000, 1D)).toMap

            val interaction3 = pubFeat.map{p=>
              impFeat.map{i=>
                p._1.toString + i._1.toString
              }
            }.flatten.map(x=> (hash(x) % 1000000 + 5000000, 1D)).toMap
*/

            val feat = userFeat ++ pubFeat ++ impFeat ++ hourFeat ++ interaction1 //++ interaction2 //++ interaction3

            val filteredFeat = feat
              .map { x =>
                val k = new Key(x._1.toString.getBytes())
                filter.add(k)
                val c = filter.approximateCount(k)
                (x, if (c >= 10) true else false)
              }.filter(x=> x._2).map(x=> x._1)

            (y.toInt, filteredFeat)
        }

      }.rdd

      /*
      val t2 = train.map{
        case Row(
        userId, dt, dateTime, id, click, c1,pubId, pubDomain, pubCategory, impression,
        pub, hourCount, dayCount
        )=>

          val y = click.asInstanceOf[String]
          //val t = topic.asInstanceOf[DenseVector]

          //val tt = t.toArray.zipWithIndex.maxBy(x=> x._1)._2

          //val topicFeat = (0 to 9).map(x=> (hash("t" + x)%100, t(x))).toMap

          val a1 = if(dayCount.asInstanceOf[Long] >= 500) userId.toString else dayCount.asInstanceOf[Long].toString
          val a2 = if(hourCount.asInstanceOf[Long] >= 30) userId.toString else hourCount.asInstanceOf[Long].toString

          val userFeat = Array(

            ("userIdCtr" + a1),
            ("userIdHctr"+ a2)
          ).map(x=> (hash(x) % 100000, 1D)).toMap


          val pubFeat = Array(
            ("pub" + pub.asInstanceOf[String])
          ).map(x=> (hash(x) % 100000 + 100000, 1D)).toMap

          val impFeat = Array(
            ("imp" + impression.asInstanceOf[String])
          ).map(x=> (hash(x) % 100000 + 200000, 1D)).toMap


          val feat = userFeat ++ pubFeat ++ impFeat

          val filter = new CountingBloomFilter(100000, 7, MURMUR_HASH)

          val filteredFeat = feat
            .map { x =>
              val k = new Key(x._1.toString.getBytes())
              filter.add(k)
              val c = filter.approximateCount(k)
              (x, if (c >= 1) true else false)
            }.filter(x=> x._2).map(x=> x._1)

          (y.toInt, filteredFeat)

      }.rdd
      */

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

      val oos = new ObjectOutputStream(new FileOutputStream("./avazu/ftrlParam"))
      oos.writeObject(model.weight)
      oos.close

    }




  }

}
