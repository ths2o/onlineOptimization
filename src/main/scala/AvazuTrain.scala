import breeze.linalg.SparseVector
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext

import org.apache.hadoop.util.bloom.CountingBloomFilter
import org.apache.hadoop.util.bloom.Key
import org.apache.hadoop.util.hash.Hash.MURMUR_HASH



/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuTrain {





  import java.io.{FileInputStream, FileOutputStream, PrintStream, ObjectInputStream, ObjectOutputStream}
  import java.util.zip.GZIPInputStream

  import scala.io.Source

  import scala.util.hashing.MurmurHash3

  /**
    *
    * id, click, hour, C1, banner_pos,
    * site_id, site_domain, site_category, app_id, app_domain,
    * app_category, device_id, device_ip, device_model, device_type,
    * device_conn_type, C14, C15, C16, C17,
    * C18, C19, C20, C21

    */

  case class data(
                   id:String, click:String, hour:String, C1:String, banner_pos:String,
                   site_id:String, site_domain:String, site_category:String,
                   app_id:String, app_domain:String,
                   app_category:String, device_id:String, device_ip:String,
                   device_model:String, device_type:String,
                   device_conn_type:String, C14:String, C15:String,
                   C16:String, C17:String, C18:String,
                   C19:String, C20:String, C21:String
                 )

  def hash(c: String) = {
    math.abs(MurmurHash3.stringHash(c, 1))
  }

  def main(args: Array[String]): Unit = {


    val sc = SparkContext.getOrCreate()
    LogManager.getRootLogger().setLevel(Level.OFF)


    val reader = new GZIPInputStream(new FileInputStream("/Users/Taehee/Downloads/train.gz"))
    //val writer = new PrintStream(new FileOutputStream("~/stream/avazu/avazu-train", true))
    //val filename = "/Users/Taehee/Downloads/test"

    var i = 1

    var dat : Array[(Int, SparseVector[Double])] = Array.empty

    val model = new FtrlSpark().setAlpha(10).setBeta(1).setL1(2).setL2(0)

    val aa = new CountingBloomFilter(100000, 7, MURMUR_HASH)



    for (line <- Source.fromInputStream(reader).getLines) {

      //println(line)

      val p = if (i == 1) Array.fill(24)("0") else line.split(",")
      val d = data(
        p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9),
        p(10), p(11), p(12), p(13), p(14), p(15), p(16), p(17), p(18), p(19),
        p(20), p(21), p(22), p(23)
      )

      import java.text.SimpleDateFormat


      val date = if (i == 1)  "20140101" else "20" + d.hour.take(6)
      val dd = new SimpleDateFormat("yyyyMMdd")
      val day = dd.parse(date).getDay()

      val hour = if (i == 1)  "01" else d.hour.drop(6)

      val (pubId, pubDomain, pubCategory) = if (d.app_id == "ecad2386"){

        (d.site_id, d.site_domain, d.site_category)
      } else {
        (d.app_id, d.app_domain, d.app_category)
      }

      val userId = if(d.device_id == "a99f214a") d.device_ip + d.device_model else d.device_id

      //println(day, hour)

      val y = d.click
      val x = Array(
        //"1" + d.hour,
        "d-" + day,
        "h-" + hour,
        "c1-" + d.C1,
        "bp-" + d.banner_pos,
        "pId-" + pubId, "pd-" + pubDomain, "pc-" + pubCategory,
        "uId-" + userId,
        "up-" + userId + pubCategory,
        "uAd-" + userId + d.C17,
        "dt-" + d.device_type,
        "dc-" + d.device_conn_type,
        "c14-" + d.C14, "c15-" + d.C15, "c16-" + d.C16, "c17-" + d.C17,
        "c18-" + d.C18, "c19-" + d.C19, "c20-" + d.C20, "c21-" + d.C21
      )
        /*
        .map { x =>
        val k = new Key(x.getBytes())
        aa.add(k)
        val c = aa.approximateCount(k)
        (x, if (c >= 14) true else false)
      }.filter(x=> x._2).map(x=> x._1)
*/


      val xHash = x.map(s => (hash(s) % 10000000, 1D)).toMap

      val sparseX = FtrlRun.mapToSparseVector(xHash, Int.MaxValue)

      dat = dat ++ Array((y.toInt, sparseX))

      i += 1

      if (i % 100000 == 0) {
        val datRdd = sc.parallelize(dat)

        model.update(datRdd)

        val summary = model.bufferSummary(0.5)
        val summaryString = Array(
          "loss : " + "%.5f".format(summary._1),
          "precision : " + "%.5f".format(summary._2),
          "AUC : " + "%.5f".format(summary._3),
          "Non-zero Coef : " + summary._4,
          "Sample Count :" + i
        )

        println(summaryString.mkString(",  "))

        dat = Array.empty


        val oos = new ObjectOutputStream(new FileOutputStream("/Users/Taehee/Downloads/ftrlParam"))
        oos.writeObject(model.weight)
        oos.close

      }




    }


      //writer.close()

    }

}
