/**
  * Created by Taehee on 2018. 5. 19..
  */


import breeze.linalg.SparseVector
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext


object FtrlRun2 {



  def mapToSparseVector(kv : Map[Int, Double], n:Int) = {
    val vec = SparseVector.zeros[Double](n)
    kv.keys.foreach(i => vec(i) = kv(i))
    vec
  }

  def rGaussian(n:Int) = {
    (1 to n).map(x=> util.Random.nextGaussian())
  }

  def rBernoulli(n:Int, p:Double) = {
    (1 to n).map(x=> if(util.Random.nextInt(10000) <= p * 10000) 1 else 0)
  }



  def sigmoid(a:Double) = {
    1 / (1 + math.exp(-a))
  }

  def logisticSample(coef:SparseVector[Double]) = {

    val sample = coef.index.map(x=> (x, rGaussian(1)(0))).toMap
    val feature = mapToSparseVector(sample, coef.length)
    val label = rBernoulli(1, sigmoid(coef.dot(feature)))(0)
    (label, feature)

  }

  def nLogisticSample(n:Int, coef:SparseVector[Double]) = {
    (1 to n).map(x=> logisticSample(coef))
  }

  def makeCoef(v:Int, r:Int) = {
    (1 to 3).map(x=> (util.Random.nextInt(r), rGaussian(1)(0) * 0.1 + v)).toMap
  }


  def linear(w : SparseVector[Double], x:SparseVector[Double]) = {
    w.dot(x)
  }



  /*
    ~/Documents/project/spark-2.3.0-bin-hadoop2.7/bin/spark-submit \
    --class FtrlRun2 ~/Documents/project/onlineOptimization/target/scala-2.11/followTheRegularizedLeader-assembly-0.1.0-SNAPSHOT.jar

    */


  def main(args: Array[String]): Unit = {




    val aa = (1 to 800).map(x=> makeCoef(5, 100)) union (1 to 200).map(x=> makeCoef(0, 100000))

    val bb = util.Random.shuffle(aa).toArray.
      map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
      flatten

    val data = bb
    //println(data.mkString("\n"))


    val sc = SparkContext.getOrCreate()
    LogManager.getRootLogger().setLevel(Level.OFF)

    val ss = sc.parallelize(bb).repartition(4)

    val tt = (0 to 1).map{x=>
      val aa = (1 to 20000).map(x=> makeCoef(5, 100)) union (1 to 10000).map(x=> makeCoef(0, 100000))
      val bb = util.Random.shuffle(aa).toArray.
        map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
        flatten
      val data = bb
      val ss = sc.parallelize(bb)
      ss
    }.toArray


    val kk = new FtrlSpark().
      setAlpha(10).
      setBeta(10).
      setL1(3).setL2(0)

    val tt1 = tt.reduce((a,b)=> a ++ b)//.repartition(20)

    kk.update(tt1)

    tt1.take(1000).map{x=>
      if (kk.predictLabel(x._2, 0.5) == x._1) 1 else 0
    }.sum.toDouble / 1000



    //kk.update(ss)

      tt.foreach{x=>
        kk.update(x)
        println(kk.nonZeroCoef)
        //println(kk.globalW.getOrElse(1, 0D), kk.globalP.getOrElse(1, 0D), kk.globalN.getOrElse(1, 0D))
      }

    //println(kk.globalW)

    val gg = new Ftrl2().
      setAlpha(10).
      setBeta(10).
      setL1(3).setL2(0).
      setW(kk.globalW).setPerCoordinateLearningRate(kk.globalP, kk.globalN)

    val pp = new Ftrl2().
      setAlpha(10).
      setBeta(10).
      setL1(3).setL2(0)

    val ll = (0 to 0).map{x=>
      val aa = (1 to 3200).map(x=> makeCoef(5, 100)) union (1 to 800).map(x=> makeCoef(0, 100000))
      val bb = util.Random.shuffle(aa).toArray.
        map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
        flatten
      val data = bb
      data
    }.toArray.flatten

    ll.foreach{x=>
      gg.update(x)
      pp.update(x)
      println(gg.bufferSummary(0.5), "    ", pp.bufferSummary(0.5))
      //println(gg.cumGradSq)
      //println(Ftrl2.cumGradSquareApprox(gg.nCount, gg.pCount, gg.gMap, x._2))
    }

    val qq= new Ftrl2().
      setAlpha(10).
      setBeta(10).
      setL1(3).setL2(0).
      setW(pp.wMap).setPerCoordinateLearningRate(pp.pCount, pp.nCount)

    ll.take(1000).foreach{x=>
      qq.update(x)
      //pp.update(x)
      println(ll.map{x=> if (qq.predictLabel(x._2, 0.5) == x._1) 1 else 0}.sum.toDouble / 4000)

    }





  }
}
