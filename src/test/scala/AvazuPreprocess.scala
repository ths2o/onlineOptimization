import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

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

    import spark.implicits._

    val aa = spark.read.option("header", "true").csv("/Users/Taehee/Downloads/train.gz")

    aa.write.partitionBy("hour").parquet("/Users/Taehee/Documents/project/avazu")

    //val bb = spark.read.parquet("/Users/Taehee/Documents/project/avazu/hour=14102100")



    val bb = spark.read.parquet("/Users/Taehee/Documents/project/avazu/hour=141021*").
      withColumn("userId", expr("case when device_id = 'a99f214a' then concat(device_ip, device_model) else device_id end")).
      withColumn("pubId", expr("case when app_id = 'ecad2386' then site_id else app_id end")).
      withColumn("pubDomain", expr("case when app_id = 'ecad2386' then site_domain else app_domain end")).
      withColumn("pubCategory", expr("case when app_id = 'ecad2386' then site_category else app_category end")).
      drop(
        "app_id", "site_id", "app_domain", "site_domain", "app_category", "site_category",
        "device_id", "device_ip", "device_model"
      )


    val cc = bb.
      select("userId", "pubId").
      groupBy("userId").agg(concat_ws(" ", collect_list("pubId")).as("pub"))


    val tokenizer = new Tokenizer().setInputCol("pub").setOutputCol("words")
    val wordsData = tokenizer.transform(cc)

    val hashingTF = new HashingTF().
      setInputCol("words").
      setOutputCol("features")

    val featurizedData = hashingTF.transform(wordsData)

    val lda = new LDA().setK(10).setMaxIter(10).setOptimizer("online")

    val model1 = lda.fit(featurizedData)

    val transformed = model1.transform(featurizedData)

    val tt = transformed.select("userId", "topicDistribution")


    tt.write.parquet("/Users/Taehee/Documents/project/avazuTopic")

    val qq = bb.join(tt, Seq("userId"))



    val ind = spark.read.parquet("/Users/Taehee/Documents/project/avazu").
      select("hour").distinct.map(x=> x.getAs[Int]("hour")).
      collect.sortBy(x=> x).map(x=> x.toString.take(6)).distinct


    val tt1 = spark.read.parquet("/Users/Taehee/Documents/project/avazuTopic")


    ind.foreach{x=>
      val ind = spark.read.parquet(s"/Users/Taehee/Documents/project/avazu").
        where(s"substr(hour, 1, 6) = $x").
        withColumn("userId", expr("case when device_id = 'a99f214a' then concat(device_ip, device_model) else device_id end")).
        withColumn("pubId", expr("case when app_id = 'ecad2386' then site_id else app_id end")).
        withColumn("pubDomain", expr("case when app_id = 'ecad2386' then site_domain else app_domain end")).
        withColumn("pubCategory", expr("case when app_id = 'ecad2386' then site_category else app_category end")).
        drop(
          "app_id", "site_id", "app_domain", "site_domain", "app_category", "site_category",
          "device_id", "device_ip", "device_model"
        )

      val joined = ind.join(tt1, Seq("userId"))
      joined.repartition(1).write.mode(SaveMode.Append).partitionBy("hour").
        parquet(s"/Users/Taehee/Documents/project/avazu2")
      println(x)
    }

    val gg = spark.read.parquet(s"/Users/Taehee/Documents/project/avazu2").
      where("substring(hour, 1, 6) = '141021'").
      withColumn("d", expr("substring(hour, 1, 6)")).
      withColumn("h", expr("substring(hour, 7, 8)"))



    def groupByCtr(data:DataFrame, c : String) ={
      data.groupBy(c).agg(count("*") as(s"${c}Ctr"))
    }

    def groupByHourlyCtr(data:DataFrame, c : String) ={
      data.groupBy(c, "h").agg(count("*") as(s"${c}Hctr"))
    }


    val ind2 = spark.read.parquet("/Users/Taehee/Documents/project/avazu").
      select("hour").distinct.map(x=> x.getAs[Int]("hour")).
      collect.sortBy(x=> x).map(x=> x.toString.take(6)).distinct

    val ind3 = (Array("a") ++ ind2).zip(ind2 ++ Array("a")).drop(1).dropRight(1)



    ind3.foreach{x=>

      val gg = spark.read.parquet(s"/Users/Taehee/Documents/project/avazu").
        where(s"substring(hour, 1, 6) = '${x._1}'").
        withColumn("d", expr("substring(hour, 1, 6)")).
        withColumn("h", expr("substring(hour, 7, 8)")).
        withColumn("userId", expr("case when device_id = 'a99f214a' then concat(device_ip, device_model) else device_id end")).
        withColumn("pubId", expr("case when app_id = 'ecad2386' then site_id else app_id end")).
        withColumn("pubDomain", expr("case when app_id = 'ecad2386' then site_domain else app_domain end")).
        withColumn("pubCategory", expr("case when app_id = 'ecad2386' then site_category else app_category end")).
        drop(
          "app_id", "site_id", "app_domain", "site_domain", "app_category", "site_category",
          "device_id", "device_ip", "device_model"
        )

      val kk = spark.read.parquet(s"/Users/Taehee/Documents/project/avazu").
        where(s"substring(hour, 1, 6) = '${x._2}'").
        withColumn("d", expr("substring(hour, 1, 6)")).
        withColumn("h", expr("substring(hour, 7, 8)")).
        withColumn("userId", expr("case when device_id = 'a99f214a' then concat(device_ip, device_model) else device_id end")).
        withColumn("pubId", expr("case when app_id = 'ecad2386' then site_id else app_id end")).
        withColumn("pubDomain", expr("case when app_id = 'ecad2386' then site_domain else app_domain end")).
        withColumn("pubCategory", expr("case when app_id = 'ecad2386' then site_category else app_category end")).
        drop(
          "app_id", "site_id", "app_domain", "site_domain", "app_category", "site_category",
          "device_id", "device_ip", "device_model"
        )

      val ss = kk.
        join(groupByCtr(gg, "userId"), Seq("userId"), "leftouter").
        join(groupByCtr(gg, "C14"), Seq("C14"), "leftouter").
        join(groupByCtr(gg, "C17"), Seq("C17"), "leftouter").
        join(groupByCtr(gg, "pubId"), Seq("pubId"), "leftouter").
        join(groupByCtr(gg, "pubCategory"), Seq("pubCategory"), "leftouter").
        join(groupByHourlyCtr(gg, "userId"), Seq("userId", "h"), "leftouter")
      //join(groupByHourlyCtr(gg, "C17"), Seq("C17", "h"), "leftouter").
      //join(groupByHourlyCtr(gg, "pubId"), Seq("pubId", "h"), "leftouter").
      //join(groupByHourlyCtr(gg, "pubCategory"), Seq("pubCategory", "h"), "leftouter")

      ss.repartition(1).write.mode(SaveMode.Append).partitionBy("hour").
        parquet(s"/Users/Taehee/Documents/project/avazu4")
      println(x)

    }






  }

}
