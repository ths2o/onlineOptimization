import breeze.linalg.SparseVector
import org.apache.spark.rdd.RDD

/**
  * Created by Taehee on 2018. 6. 2..
  */

class FtrlSpark {

  var globalP = SparseVector.zeros[Double](Int.MaxValue)
  var globalN = SparseVector.zeros[Double](Int.MaxValue)
  var globalW = SparseVector.zeros[Double](Int.MaxValue)

  def update(data:RDD[(Int, SparseVector[Double])]) ={
    val model = FtrlSpark.ftrlPar(data, this.globalP, this.globalN, this.globalW)
    this.globalP = model._1
    this.globalN = model._2
    this.globalW = model._3
    this
  }


}

object FtrlSpark {


  def ftrlPar(
               data : RDD[(Int, SparseVector[Double])],
               globalP : SparseVector[Double],
               globalN : SparseVector[Double],
               globalW : SparseVector[Double]

             ) = {

    //val numPartitions = 4
    //val numParts = if (numPartitions > 0) numPartitions else data.getNumPartitions
    //val context = data.context

    val w = data.mapPartitions{ part =>
      var localW = globalW
      var localP = globalP
      var localN = globalN
      var internalP = SparseVector.zeros[Double](Int.MaxValue)
      var internalN = SparseVector.zeros[Double](Int.MaxValue)

      val updater = new Ftrl2().setAlpha(5).setBeta(1).setL1(1.5).setL2(0)
      updater.cumGradSq = Ftrl2.cumGradSquareApprox(localN, localP)
      updater.weight = localW

      part.foreach{x=>

        updater.update(x)

        if (x._1 == 1) {
          localP = Ftrl2.counter(localP, x._2)
          internalP = Ftrl2.counter(internalP, x._2)
        } else {
          localN = Ftrl2.counter(localN, x._2)
          internalN = Ftrl2.counter(internalN, x._2)
        }

        updater.cumGradSq = Ftrl2.cumGradSquareApprox(localN, localP)
      }

      val normalizeFactor = SparseVector.zeros[Double](Int.MaxValue)
      val aa = (internalN + internalP)
      aa.compact()
      aa.activeIterator.foreach{x=>
        normalizeFactor.update(x._1, 1)
      }

      Iterator((updater.weight, internalN, internalP, normalizeFactor))
    }.reduce{case ((a1, a2, a3, a4), (b1, b2, b3, b4)) =>
      (a1 + b1, a2 + b2, a3 + b3, a4 + b4)
    }

    (w._1 /:/ w._4, w._2, w._3)

  }





  def main(args: Array[String]): Unit = {

  }
}
