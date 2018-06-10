import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.expressions.Window
import scala.util.hashing.MurmurHash3


/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuPreprocess2 {






  def hash(c: String) = {
    math.abs(MurmurHash3.stringHash(c, 1))
  }

  def main(args: Array[String]): Unit = {


    val sc = SparkContext.getOrCreate()

    val spark = SparkSession.builder().getOrCreate()

    import spark.implicits._

//    val aa = spark.read.option("header", "true").csv("/Users/Taehee/Downloads/train.gz")


    val aa = spark.read.option("header", "true").csv("./stream/avazu/train.gz")


    aa.write.partitionBy("hour").parquet("./avazu/train1")

    val test = spark.read.option("header", "true").csv("./stream/avazu/test.gz")
    test.write.partitionBy("hour").parquet("./avazu/test1")

    //val bb = spark.read.parquet("/Users/Taehee/Documents/project/avazu/hour=14102100")


    //val train = spark.read.parquet("/Users/Taehee/Documents/project/avazu")


    val t1 = spark.read.parquet("./avazu/train1")

    val t2 = spark.read.parquet("./avazu/test1").
      withColumn("click", lit("1")).select(t1.columns.map(col): _*)

    val train = t1.union(t2).
      withColumn("userId", expr("case when device_id = 'a99f214a' then concat(device_ip, device_model) else device_id end")).
      withColumn("pubId", expr("case when app_id = 'ecad2386' then site_id else app_id end")).
      withColumn("pubDomain", expr("case when app_id = 'ecad2386' then site_domain else app_domain end")).
      withColumn("pubCategory", expr("case when app_id = 'ecad2386' then site_category else app_category end")).
      drop(
        "app_id", "site_id", "app_domain", "site_domain", "app_category", "site_category",
        "device_id", "device_ip", "device_model"
      )


    val appBag = train.
      select("userId", "pubId").distinct().
      groupBy("userId").agg(concat_ws(" ", collect_list("pubId")).as("pub"))


    val imp = train.
      withColumn("impression",
        expr("concat(banner_pos, device_type, device_conn_type, C14, C15, C16, C17, C18, C19, C20, C21)")).
      withColumn("y", expr("concat('20', substring(hour, 1, 2))")).
      withColumn("mo", expr("substring(hour, 3, 2)")).
      withColumn("d", expr("substring(hour, 5, 2)")).
      withColumn("h", expr("substring(hour, 7, 2)")).
      withColumn("dateTime", expr("concat(y, '-', mo, '-', d, ' ', h, ':00:00')")).
      withColumn("dateTime", to_timestamp(col("dateTime"), "yyyy-MM-dd HH:mm:ss")).
      withColumn("dt", expr("dateTime").cast(DateType)).
      drop("banner_pos", "device_type", "device_conn_type",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "y", "mo", "d", "h")



    val userHCount = train.groupBy("userId", "hour").agg(count("*").as("hourCount")).
      withColumn("y", expr("concat('20', substring(hour, 1, 2))")).
      withColumn("mo", expr("substring(hour, 3, 2)")).
      withColumn("d", expr("substring(hour, 5, 2)")).
      withColumn("h", expr("substring(hour, 7, 2)")).
      withColumn("dateTime", expr("concat(y, '-', mo, '-', d, ' ', h, ':00:00')")).
      withColumn("dateTime", to_timestamp(col("dateTime"), "yyyy-MM-dd HH:mm:ss")).
      withColumn("aa", window(col("dateTime"), "1 hours")).
      withColumn("start", expr("aa.start")).withColumn("lastHour", expr("aa.end")).
      drop("aa", "y", "mo", "d", "h", "start", "dateTime", "hour").
      withColumnRenamed("lastHour", "dateTime")

    val userDCount = train.select("userId", "hour").
      withColumn("y", expr("concat('20', substring(hour, 1, 2))")).
      withColumn("mo", expr("substring(hour, 3, 2)")).
      withColumn("d", expr("substring(hour, 5, 2)")).
      withColumn("h", expr("substring(hour, 7, 2)")).
      withColumn("dateTime", expr("concat(y, '-', mo, '-', d)")).
      withColumn("dateTime", expr("dateTime").cast(DateType)).
      withColumn("lastDay", expr("date_add(dateTime, 1)")).
      groupBy("userId", "lastDay").agg(count("*").as("dayCount")).
      withColumnRenamed("lastDay", "dt")


    val train2 = imp.join(appBag, Seq("userId"), "leftouter").
      join(userHCount, Seq("userId", "dateTime"), "leftouter").
      join(userDCount, Seq("userId", "dt"), "leftouter")


    train2.repartition(1).write.partitionBy("hour").parquet("./avazu/trainTest")

    val tt = spark.read.parquet("./avazu/trainTest")


  }

}
