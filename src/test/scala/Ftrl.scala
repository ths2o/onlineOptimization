/**
  * Created by Taehee on 2018. 5. 19..
  */


import breeze.linalg.SparseVector

case class FtrlParam(
                    n: Int, i: Int,
                    alpha:Double, beta:Double, lambda:Double, lambda2:Double,
                    weight : SparseVector[Double], z : SparseVector[Double],
                    eta : SparseVector[Double], cumGrad : SparseVector[Double],
                    cumGradSq : SparseVector[Double], nonZeroCoef : Int,
                    bufferSize:Int, buffer: Array[(Int, Double, Double, Array[Int])]
                    )

class Ftrl {

  var n = Int.MaxValue
  var i = 1
  var alpha:Double = 5.0
  var beta:Double = 1.0
  var lambda:Double = 0.0
  var lambda2:Double = 0.0
  var weight : SparseVector[Double] = SparseVector.zeros[Double](n)
  var z : SparseVector[Double] = SparseVector.zeros[Double](n)
  var eta : SparseVector[Double] = SparseVector.zeros[Double](n)
  var cumGrad : SparseVector[Double] = SparseVector.zeros[Double](n)
  var cumGradSq : SparseVector[Double] = SparseVector.zeros[Double](n)
  var nonZeroCoef : Int = 0
  var bufferSize = 1000
  var buffer : Array[(Int, Double, Double, Array[Int])] =Array.empty

  def load(param:FtrlParam) = {
    this.n = param.n
    this.i = param.i
    this.alpha = param.alpha
    this.beta = param.beta
    this.lambda = param.lambda
    this.lambda2 = param.lambda2
    this.weight = param.weight
    this.z = param.z
    this.eta = param.eta
    this.cumGrad = param.cumGrad
    this.cumGradSq =  param.cumGradSq
    this.nonZeroCoef = param.nonZeroCoef
    this.bufferSize = param.bufferSize
    this.buffer = param.buffer
    this
  }

  def save() = {
    FtrlParam(
      this.n,
      this.i,
      this.alpha,
      this.beta ,
      this.lambda,
      this.lambda2,
      this.weight,
      this.z,
      this.eta,
      this.cumGrad,
      this.cumGradSq,
      this.nonZeroCoef,
      this.bufferSize,
      this.buffer
    )
  }

  def setVectorSize(n:Int) = {
    this.n = n
    this
  }

  def setAlpha (alpha:Double) = {
    this.alpha = alpha
    this
  }

  def setBeta (beta:Double) = {
    this.beta = alpha
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

  def update(data : (Int, SparseVector[Double])) = {

    val fit = fitStat(data)
    val updatedParam = Ftrl.ftrlUpdate(data, alpha, beta, lambda, lambda2, weight, z, eta, cumGrad, cumGradSq)
    this.weight = updatedParam._1
    this.cumGrad = updatedParam._3
    this.cumGradSq = updatedParam._4
    this.eta = updatedParam._5
    this.z = updatedParam._6
    this.i += 1
    this.nonZeroCoef = this.weight.activeSize
    this.buffer = if (this.buffer.size < this.bufferSize) this.buffer :+ fit else this.buffer.drop(1) :+ fit
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

  def bufferSummary(threshold : Double) ={
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

object Ftrl {


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

  def eta(cumGradSq : SparseVector[Double], alpha:Double, beta:Double, lambda2:Double) = {
    val iCumGradSq = cumGradSq.copy
    //iCumGradSq.activeIterator.foreach(x=> iCumGradSq.update(x._1, alpha/(beta + math.sqrt(x._2))))
    iCumGradSq.activeIterator.foreach(x=> iCumGradSq.update(x._1, 1/((beta + math.sqrt(x._2)) / alpha + lambda2) ))
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

  def ftrlUpdate(
                       data : (Int, SparseVector[Double]),
                       alpha:Double,
                       beta:Double,
                       lambda:Double,
                       lambda2:Double,
                       oldWeight : SparseVector[Double],
                       oldZ : SparseVector[Double],
                       oldEta : SparseVector[Double],
                       oldCumGrad : SparseVector[Double],
                       oldCumGradSq : SparseVector[Double]
                     ) = {


    val weight = wUpdater(oldZ, oldEta, lambda); weight.compact()
    val grad = gradientLL(data._1, weight, data._2); grad.compact()
    val cumGrad = oldCumGrad + grad; cumGrad.compact()
    val cumGradSq = oldCumGradSq + grad *:* grad; cumGradSq.compact()
    val newEta = eta(cumGradSq, alpha, beta, lambda2); newEta.compact
    val newZ = zUpdater(oldZ, grad, oldEta, newEta, weight); newZ.compact()

    (weight, grad, cumGrad, cumGradSq, newEta, newZ)

  }
}

