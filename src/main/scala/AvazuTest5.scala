import breeze.linalg.SparseVector
import org.apache.hadoop.util.bloom.{CountingBloomFilter, Key}
import org.apache.hadoop.util.hash.Hash._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._

/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuTest5 {


  import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, PrintStream}
  import java.util.zip.GZIPInputStream

  import scala.io.Source
  import scala.util.hashing.MurmurHash3


  def hash(c: String) = {
    math.abs(MurmurHash3.stringHash(c, 1))
  }

  def main(args: Array[String]): Unit = {



/*

    ~/spark-2.3.0-bin-hadoop2.7/bin/spark-submit \
    --class AvazuTest5 ~/project/onlineOptimization/target/scala-2.11/followTheRegularizedLeader-assembly-0.1.0-SNAPSHOT.jar

 */



    val sc = SparkContext.getOrCreate()
    LogManager.getRootLogger().setLevel(Level.OFF)

    val spark = SparkSession.builder().getOrCreate()


    import spark.implicits._


    val train = spark.read.parquet(s"./avazu/trainTest/hour=141031*").
      withColumn("hourCount", expr("case when hourCount is null then 0 else hourCount end")).
      withColumn("dayCount", expr("case when dayCount is null then 0 else dayCount end")).
      withColumn("h", expr("substring(dateTime, 12, 2)"))

    val t2 = train.mapPartitions { part =>

      part.map {
        case Row(
        userId, dt, dateTime, id, click, c1, pubId, pubDomain, pubCategory, impression,
        pub, hourCount, dayCount, h
        ) =>

          val y = click.asInstanceOf[String]

          val a1 = if (dayCount.asInstanceOf[Long] >= 500) userId.toString else dayCount.asInstanceOf[Long].toString
          val a2 = if (hourCount.asInstanceOf[Long] >= 30) userId.toString else hourCount.asInstanceOf[Long].toString

          val userFeat = Array(

            ("userIdCtr" + a1),
            ("userIdHctr"+ a2)
          ).map(x=> (hash(x), 1D)).toMap


          val pubFeat = Array(
            ("pub" + pub.asInstanceOf[String])
          ).map(x=> (hash(x), 1D)).toMap

          val impFeat = Array(
            ("imp" + impression.asInstanceOf[String])
          ).map(x=> (hash(x), 1D)).toMap

          val hourFeat = Array(
            ("hour" + h.asInstanceOf[String])
          ).map(x=> (hash(x), 1D)).toMap


          val interaction1 = userFeat.map{u=>
            u._1.toString + h.toString
          }.map(x=> (hash(x), 1D)).toMap


          val interaction2 = pubFeat.map{p=>
            p.toString + h.toString
          }.map(x=> (hash(x), 1D)).toMap


          val interaction3 = userFeat.map{u=>
            pubFeat.map{p=>
              u._1.toString + p._1.toString
            }
          }.flatten.map(x=> (hash(x), 1D)).toMap
          /*
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

          val feat = userFeat ++ pubFeat ++ impFeat ++ hourFeat ++ interaction1 ++ interaction2 ++ interaction3

          val filteredFeat = feat

          (y.toInt, filteredFeat, id.toString)
      }

    }.rdd

    val t3 = t2.map(x => (x._1, FtrlRun.mapToSparseVector(x._2, Int.MaxValue), x._3))

    val ois = new ObjectInputStream(new FileInputStream("./avazu/ftrlParam"))
    val param1 = ois.readObject.asInstanceOf[SparseVector[Double]]
    val param = param1.copy


    val prob = t3.map{x=>

      val prob = Ftrl.p(param, x._2)

      (x._3, prob)

    }

    prob.map(x=> x._1 + "," + "%.8f".format(x._2)).repartition(1).saveAsTextFile("./avazu/sthSubmission")
/*
    val writer = new PrintStream(new FileOutputStream("./avazu/sthSubmission", true))

    writer.append("id, click\n")
    prob.foreach{x=>
      val output = x._1 + "," + "%.8f".format(x._2)
      writer.append(output + "\n")
    }

    writer.close()
*/




  }
}