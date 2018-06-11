
/**
  * Created by Taehee on 2018. 5. 19..
  */

import breeze.linalg.SparseVector

class Ogd {

  var n = Int.MaxValue /*2,147,483,647*/
  var weight : SparseVector[Double] = SparseVector.zeros[Double](n)
  var i = 1


  def setVectorSize(n:Int) = {
    this.n = n
    this
  }

  def update(data : (Int, SparseVector[Double])) = {
    val updatedParam = Ogd.ogdUpdate(data, i, weight)
    this.weight = updatedParam._1
    this.i += 1
    this.weight
  }

  def predictProb (data : SparseVector[Double]) = {
    Ftrl.sigmoid(weight.dot(data))
  }

  def predictLabel (data : SparseVector[Double], threshold:Double) ={
    if (predictProb(data) >= threshold) 1 else 0
  }


}

object Ogd {


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

  def ogdUpdate(
                       data : (Int, SparseVector[Double]),
                       t : Double,
                       oldWeight : SparseVector[Double]

                     ) = {

    val weight = gradientUpdater(data._1, oldWeight, data._2, 1/math.sqrt(t))
    (weight, t)
  }

}

