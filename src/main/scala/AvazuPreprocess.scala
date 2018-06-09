import breeze.linalg.SparseVector
import org.apache.hadoop.util.bloom.{CountingBloomFilter, Key}
import org.apache.hadoop.util.hash.Hash.MURMUR_HASH
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import java.io.{FileInputStream, FileOutputStream, ObjectOutputStream}
import java.util.zip.GZIPInputStream

import scala.io.Source
import scala.util.hashing.MurmurHash3


/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuPreprocess {






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
