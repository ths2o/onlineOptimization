import breeze.linalg.SparseVector
import org.apache.hadoop.util.bloom.{CountingBloomFilter, Key}
import org.apache.hadoop.util.hash.Hash.MURMUR_HASH
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession


/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuPreprocess {





  import java.io.{FileInputStream, FileOutputStream, ObjectOutputStream}
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

    val spark = SparkSession.builder().getOrCreate()


    val aa = spark.read.option("header", "true").csv("/Users/Taehee/Downloads/train.gz")

    aa.write.partitionBy("hour").parquet("/Users/Taehee/Documents/project/avazu")

    val bb = spark.read.parquet("/Users/Taehee/Documents/project/avazu/hour=14102100")





    }

}
