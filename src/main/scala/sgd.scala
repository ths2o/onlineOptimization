/**
  * Created by Taehee on 2018. 5. 19..
  */


object sgd {

  val data: Array[(Int, Vector[Double])] = Array(
    (1, Vector(1.0, 2.0, 1.0)), (1, Vector(1.0, 3.0, 2.0)), (1, Vector(1.0, 4.0, 5.0)),
    (0, Vector(1.0, 1.1, 0.2)), (0, Vector(1.0, 2.2, 0.4)), (0, Vector(1.0, 2.1, 1.7))
  )


  val n = 100

  def rGaussian(n:Int) = {
    (1 to n).map(x=> util.Random.nextGaussian())
  }

  def rBernoulli(n:Int, p:Double) = {
    (1 to n).map(x=> if(util.Random.nextInt(10000) <= p * 10000) 1 else 0)
  }

  def logisticSample(coef:Array[Double]) = {

    val feature = coef.map{x=> val ran = rGaussian(1)(0); (ran, x*ran)}
    val label = rBernoulli(1, sigmoid(feature.map(x=> x._2).sum))(0)
    (label, feature.map(x=> x._1).toVector)

  }

  def nLogisticSample(n:Int, coef:Array[Double]) = {
    (1 to n).map(x=> logisticSample(coef))
  }


  def linear(w : Vector[Double], x:Vector[Double]) = {
    w.zip(x).map{case (weight, data)=> weight * data}.sum
  }

  def sigmoid(a:Double) = {
    1 / (1 + math.exp(-a))
  }

  def p(w : Vector[Double], x:Vector[Double]) = {
    val sig = sigmoid(linear(w, x))
    if (sig == 0) 0.0000000000001 else if (sig == 1) 0.999999999999 else sig
  }

  def logLoss(y : Int, w : Vector[Double], x:Vector[Double]) = {
    -y * math.log(p(w, x)) - (1 - y) * math.log(1 - p(w, x))
  }

  def gradientLL(y : Int, w : Vector[Double], x:Vector[Double]) = {
    val diff = p(w, x) - y
    x.map(data=> diff * data)
  }

  def gradientUpdater(y : Int, w : Vector[Double], x:Vector[Double], learningRate:Double) ={
    val grad = gradientLL(y, w, x)
    w.zip(grad.map(_ * learningRate)).map(k => k._1 - k._2)
  }


  def gradientDescent(
                       data : Array[(Int, Vector[Double])],
                       learnRate : Double,
                       decay : Double,
                       initialWeight : Vector[Double],
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
      util.Random.shuffle(data.toList).foreach { t =>
        weight = gradientUpdater(t._1, weight, t._2, learningRate)
      }

      newLoss = data.map{d=> logLoss(d._1, weight, d._2)}.sum
      learningRate = learningRate * decay
      //println(data.map{d=> (logLoss(d._1, weight, d._2), p(weight, d._2))}.mkString("\n"))
      println(weight, newLoss, i)
      i += 1

    }
  }



  def main(args: Array[String]): Unit = {



    val sampledData = nLogisticSample(1000, Array(1,2,3,4,5)).toArray

    println(sampledData.mkString("\n"))

    //val initialWeight = rGaussian(sampledData(0)._2.size).toVector

    val initialWeight = Vector(0.01, 0.01, 0.01)
    gradientDescent(data, 0.9, 0.99, initialWeight, 1000, 0.0000000001)

    //println(initialWeight)
  }
}
