

/**
  * Created by Taehee on 2018. 5. 19..
  */


import breeze.linalg.SparseVector


object sgdSparse {


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


  def gradientDescent(
                       data : Array[(Int, SparseVector[Double])],
                       learnRate : Double,
                       decay : Double,
                       initialWeight : SparseVector[Double],
                       maxEpoch : Int,
                       tol : Double

                     ) = {

    var learningRate = learnRate
    var weight = initialWeight
    var initialLoss = data.map{d=> logLoss(d._1, weight, d._2)}.sum
    var (oldLoss, newLoss) = (0D, initialLoss)

    var i = 1

    while (math.abs(newLoss - oldLoss) > tol && i <= maxEpoch) {

      oldLoss = newLoss
      /*util.Random.shuffle(data.toList)*/data.foreach { t =>
        weight = gradientUpdater(t._1, weight, t._2, learningRate)
        //newLoss = data.map{d=> logLoss(d._1, weight, d._2)}.sum

        i += 1
        if (i % 1000 == 0) println(weight, newLoss, i)
        learningRate = 1/math.sqrt(i)
        //println(data.map{d=> (logLoss(d._1, weight, d._2), p(weight, d._2))}.mkString("\n"))

      }

    }
  }



  def main(args: Array[String]): Unit = {



    val coef = Map(0 -> 1.0, 1-> 2.0, 2->3.0, 3-> 4.0, 4-> 5.0, 5-> 0D, 9->0D)

    val coef2 = Map(0 -> 1.0, 1-> 2.0, 2->3.0, 3-> 4.0, 4-> 5.0, 6-> 15D, 8->7D)

    println(mapToSparseVector(coef, coef.keys.max + 1))//.index.max)
    val sampledData = nLogisticSample(500000, mapToSparseVector(coef, coef.keys.max + 1)).toArray
    val sampledData2 = nLogisticSample(500000, mapToSparseVector(coef2, coef.keys.max + 1)).toArray
    val data =sampledData.union(sampledData2)
    //println(data.mkString("\n"))


    val initialWeight = SparseVector.zeros[Double](coef.keys.max + 1)
    gradientDescent(data, 0.9, 1, initialWeight, 1, 0.0000000001)
  }
}
