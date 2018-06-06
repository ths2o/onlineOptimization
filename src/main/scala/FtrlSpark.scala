import breeze.linalg.SparseVector
import org.apache.spark.rdd.RDD

/**
  * Created by Taehee on 2018. 6. 2..
  */

class FtrlSpark {

  var globalP : Map[Int, Double] = Map.empty
  var globalN : Map[Int, Double] = Map.empty
  var globalW : Map[Int, Double] = Map.empty
  var globalZ : Map[Int, Double] = Map.empty

  var n = Int.MaxValue
  var i = 1
  var alpha:Double = 5.0
  var beta:Double = 1.0
  var lambda:Double = 0.0
  var lambda2:Double = 0.0
  var weight : SparseVector[Double] = SparseVector.zeros[Double](n)
  var nonZeroCoef : Int = 0
  var bufferSize = 1000
  var buffer : Array[(Int, Double, Double, Array[Int])] =Array.empty


  def setAlpha (alpha:Double) = {
    this.alpha = alpha
    this
  }

  def setBeta (beta:Double) = {
    this.beta = beta
    this
  }

  def setL1 (lambda:Double) = {
    this.lambda = lambda
    this
  }

  def setL2 (lambda2:Double) = {
    this.lambda2 = lambda2
    this
  }

  def update(data:RDD[(Int, SparseVector[Double])]) ={

    val fit = data.sample(false, 0.01).collect.map(x=> fitStat(x))
    //this.buffer = data.map{x=> fitStat(x)}.collect
    val model = FtrlSpark.ftrlPar(data, this.globalP, this.globalN, this.globalW, this.globalZ,
      alpha, beta, lambda, lambda2)
    //this.globalW = model._1
    //this.globalP = model._2
    //this.globalN = model._3

    this.globalW = this.globalW ++ model._1
    this.globalP = (globalP.toSeq ++ model._2.toSeq).groupBy(x=> x._1).map(x=> (x._1, x._2.map(x=> x._2).sum))
    this.globalN = (globalP.toSeq ++ model._3.toSeq).groupBy(x=> x._1).map(x=> (x._1, x._2.map(x=> x._2).sum))
    this.globalZ = this.globalZ ++ model._4
    this.weight = FtrlRun.mapToSparseVector(globalW, n)

    //println(globalW.getOrElse(1, 0D), globalP.getOrElse(1, 0D), globalN.getOrElse(1, 0D), i)

    this.i = fit.size
    this.nonZeroCoef = this.weight.activeSize
    //this.buffer = if (this.buffer.size < this.bufferSize) this.buffer :+ fit else this.buffer.drop(1) :+ fit
    this.buffer = fit


    this
  }


  def predictProb (data : SparseVector[Double]) = {
    Ftrl.sigmoid(weight.dot(data))
  }

  def predictLabel (data : SparseVector[Double], threshold:Double) ={
    if (predictProb(data) >= threshold) 1 else 0
  }


  def fitStat (data : (Int, SparseVector[Double])) = {
    val prob = predictProb(data._2)
    val loss = Ftrl.logLoss(data._1, this.weight, data._2)
    val positive = (0 to 10).map(x=> if (prob >= x.toDouble/10D) 1 else 0).toArray
    (data._1, prob, loss, positive)
  }

  def fitRdd (data : RDD[(Int, SparseVector[Double])]) = {
    data.map(x=> fitStat(x))
  }



  def bufferSummary(threshold : Double) ={
    //val buffer = fitRdd(data)
    val size = buffer.size.toDouble
    val loss = buffer.map(x=> x._3).sum / size
    val pLabel = buffer.map(x=> if (x._2 >= threshold) 1 else 0)
    val precision = buffer.map(x=> x._1).zip(pLabel).map(x=> if(x._1 == x._2) 1 else 0).sum.toDouble / size

    val roc = buffer.groupBy(_._1).map{x=>
      val aa = x._2.map(x=> x._4).reduce((a, b) => a.zip(b).map(x=> x._1 + x._2))
      val bb = x._2.size
      val cc = aa.map(t => t.toDouble/bb.toDouble)
      (x._1, cc)
    }
    val height = roc.getOrElse(1, Array.fill(11)(0D)).drop(1).zip(roc.getOrElse(1, Array.fill(11)(0D)).dropRight(1)).map(x=> (x._1 + x._2)/2)
    val width = roc.getOrElse(0, Array.fill(11)(0D)).dropRight(1).zip(roc.getOrElse(0, Array.fill(11)(0D)).drop(1)).map(x=> (x._1 - x._2))
    val auc = height.zip(width).map(x=> x._1 * x._2).sum


    (loss, precision, auc, this.nonZeroCoef)
  }


}

object FtrlSpark {



  def ftrlPar(
               data : RDD[(Int, SparseVector[Double])],
               globalP : Map[Int, Double],
               globalN : Map[Int, Double],
               globalW : Map[Int, Double],
               globalZ : Map[Int, Double],
               alpha:Double, beta:Double, lambda:Double, lambda2:Double

             ) = {

    //val numPartitions = 4
    //val numParts = if (numPartitions > 0) numPartitions else data.getNumPartitions
    //val context = data.context

    val w = data.repartition(4).mapPartitions{ part =>

      var localP :Map[Int, Double] = Map.empty
      var localN :Map[Int, Double] = Map.empty


      val updater = new Ftrl2().
        setAlpha(alpha).
        setBeta(beta).
        setL1(lambda).setL2(lambda2).
        setW(globalW).setPerCoordinateLearningRate(localP,localN).setZ(globalZ)

      part.toArray.foreach{x=>

        updater.update(x)
        if(x._1 == 1) localP = Ftrl2.counter(localP, x._2) else localN = Ftrl2.counter(localN, x._2)
        //println(updater.weight, updater.i)

        val p = if (Ftrl2.p(updater.weight, x._2) >= 0.5) 1 else 0

      //  correct += (if (x._1 == p) 1 else 0)
      //  logloss += (Ftrl2.logLoss(x._1, updater.weight, x._2))
      //  count += 1
      }

      //println(correct.toDouble/count.toDouble, logloss/count, count)
      Iterator((updater.wMap, localP, localN, updater.zMap))

    }.reduce { case ((a1, a2, a3, a4), (b1, b2, b3, b4)) =>
      val a = (a1.toSeq ++ b1.toSeq).groupBy(x=> x._1).map{x=>
        val sum = x._2.map(x=> x._2).sum
        val count = x._2.map(x=> x._2).size
        (x._1, sum/count)
      }
      val b = (a2.toSeq ++ b2.toSeq).groupBy(x=> x._1).map(x=> (x._1, x._2.map(x=> x._2).sum))
      val c = (a3.toSeq ++ b3.toSeq).groupBy(x=> x._1).map(x=> (x._1, x._2.map(x=> x._2).sum))

      val d = (a4.toSeq ++ b4.toSeq).groupBy(x=> x._1).map{x=>
        val sum = x._2.map(x=> x._2).sum
        val count = x._2.map(x=> x._2).size
        (x._1, sum/count)
      }

      (a, b, c, d)
    }


    //  reduce{case ((a1, a2, a3), (b1, b2, b3)) =>

    //}
    //(w._1 /:/ w._4, w._2, w._3)
    //println(w._1)
    w



  }





  def main(args: Array[String]): Unit = {

  }
}
