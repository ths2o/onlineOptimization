package scala



/**
  * Created by Taehee on 2018. 5. 19..
  */


import breeze.linalg.SparseVector


object Ftrl {


  val data2: Array[(Int, Map[Int, Double])] = Array(
    (1, Map(1-> 1.0, 2->2.0, 3-> 1.0)),
    (1, Map(1-> 1.0, 2->3.0, 3-> 2.0)),
    (1, Map(1-> 1.0, 2->4.0, 3-> 5.0)),
    (0, Map(1-> 1.0, 2->1.1, 3-> 0.2)),
    (0, Map(1-> 1.0, 2->2.2, 3-> 0.4)),
    (0, Map(1-> 1.0, 2->2.1, 3-> 1.7))
  )


  def mapToSparseVector(kv : Map[Int, Double], n:Int) = {
    val vec = SparseVector.zeros[Double](n)
    kv.keys.foreach(i => vec(i) = kv(i))
    vec
  }

  val data3 = data2.map(x=> (x._1, mapToSparseVector(x._2, 10)))

  val n = 100

  def rGaussian(n:Int) = {
    (1 to n).map(x=> util.Random.nextGaussian())
  }

  def rBernoulli(n:Int, p:Double) = {
    (1 to n).map(x=> if(util.Random.nextInt(10000) <= p * 10000) 1 else 0)
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



  def linear(w : SparseVector[Double], x:SparseVector[Double]) = {
    w.dot(x)
  }


  def sigmoid(a:Double) = {
    1 / (1 + math.exp(-a))
  }

  def p(w : SparseVector[Double], x:SparseVector[Double]) = {
    val sig = sigmoid(linear(w, x))
    if (sig == 0) 0.0000000000001 else if (sig == 1) 0.999999999999 else sig
  }

  def logLoss(y : Int, w : SparseVector[Double], x:SparseVector[Double]) = {
    -y * math.log(p(w, x)) - (1 - y) * math.log(1 - p(w, x))
  }

  def gradientLL(y : Int, w : SparseVector[Double], x:SparseVector[Double]) = {
    val diff = p(w, x) - y
    x * diff
    //x.map(data=> diff * data)
  }

  def gradientUpdater(y : Int,w : SparseVector[Double], x:SparseVector[Double], learningRate:Double) ={
    val grad = gradientLL(y, w, x)
    w - grad * learningRate
    //w.zip(grad.map(_ * learningRate)).map(k => k._1 - k._2)
  }

  def eta(cumGradSq : SparseVector[Double], alpha:Double, beta:Double) = {
    val iCumGradSq = cumGradSq.copy
    iCumGradSq.activeIterator.foreach(x=> iCumGradSq.update(x._1, alpha/(beta + math.sqrt(x._2))))
    iCumGradSq
  }

  def zUpdater(
                oldZ : SparseVector[Double],
                grad : SparseVector[Double],
                oldEta : SparseVector[Double],
                newEta : SparseVector[Double],
                w : SparseVector[Double]
              ) ={

    val iNewEta = newEta.copy
    iNewEta.activeIterator.foreach(x=> iNewEta.update(x._1, 1/x._2))
    val iOldEta = oldEta.copy
    iOldEta.activeIterator.foreach(x=> iOldEta.update(x._1, 1/x._2))

    oldZ + grad + (iNewEta - iOldEta) *:* w
  }

  def wUpdater(z:SparseVector[Double], eta:SparseVector[Double], lambda:Double) = {

    val newZ = z.copy
    newZ.activeIterator.foreach{x=>
      if (math.abs(x._2) <= lambda) newZ.update(x._1, 0)
      else {
        //val zi = newZ.valueAt(x._1)
        newZ.update(x._1, if(x._2 >=0) x._2 -lambda else x._2+lambda)
        //if (zi >= 0) newZ.update(x._1, zi - lambda) else newZ.update(x._1, zi + lambda)
      }
    }

    newZ.compact()
    val newW = - eta *:* newZ
    newW
  }


  def gradientDescent(
                       data : Array[(Int, SparseVector[Double])],
                       alpha:Double,
                       beta:Double,
                       lambda:Double,
                       initialWeight : SparseVector[Double],
                       maxEpoch : Int,
                       tol : Double

                     ) = {


    var weight = initialWeight
    var initialLoss = data.map{d=> logLoss(d._1, weight, d._2)}.sum
    var (oldLoss, newLoss) = (0D, initialLoss)

    var (oldZ, newZ) = (SparseVector.zeros[Double](10), SparseVector.zeros[Double](10))
    var (cumGrad, grad) = (SparseVector.zeros[Double](10), SparseVector.zeros[Double](10))
    var cumGradSq = SparseVector.zeros[Double](10)
    var (oldLRate, newLRate) = (eta(cumGradSq, alpha, beta), eta(cumGradSq, alpha, beta))
    var i = 1

    while (i <= maxEpoch) {

      oldLoss = newLoss
      /*util.Random.shuffle(data.toList)*/data.foreach { t =>

        oldZ = newZ
        oldLRate = newLRate
        weight = wUpdater(newZ, newLRate, lambda)
        grad = gradientLL(t._1, weight, t._2)

        cumGrad = cumGrad + grad
        cumGradSq = cumGradSq + grad *:* grad
        newLRate = eta(cumGradSq, alpha, beta)
        newZ = zUpdater(oldZ, grad, oldLRate, newLRate, weight)

        if (i % 1000 == 0) println(weight, i)
        i += 1


        //println(data.map{d=> (logLoss(d._1, weight, d._2), p(weight, d._2))}.mkString("\n"))

      }

    }
  }



  def main(args: Array[String]): Unit = {


    val coef = Map(0 -> 1.0, 1-> 2.0, 2->3.0, 3-> 4.0, 4-> 5.0, 5-> 0D, 9->0D)

    val coef2 = Map(0 -> 1.0, 1-> 2.0, 2->3.0, 3-> 4.0, 4-> 5.0, 6-> 15D, 8->7D)

    println(mapToSparseVector(coef, coef.keys.max + 1))//.index.max)
    val sampledData = nLogisticSample(200000, mapToSparseVector(coef, coef.keys.max + 1)).toArray
    val sampledData2 = nLogisticSample(200000, mapToSparseVector(coef2, coef.keys.max + 1)).toArray
    val data =sampledData.union(sampledData2)
    //println(data.mkString("\n"))


    val initialWeight = SparseVector.zeros[Double](coef.keys.max + 1)
    gradientDescent(data, 1, 0, 2, initialWeight, 1, 0.00001)




    /*
    val coef = Map(0 -> 1.0, 1-> 2.0, 2->3.0, 3-> 4.0, 4-> 5.0, 5-> 0D, 9->0D)
    val aa = mapToSparseVector(coef, coef.keys.max + 1)

    val cc = mapToSparseVector(coef, coef.keys.max + 1)

    val bb = wUpdater(aa, cc, 2)


    //cc.activeIterator.foreach(x=> cc.update(x._1, 1/x._2))

    println(aa, bb, cc)
*/

  }
}
