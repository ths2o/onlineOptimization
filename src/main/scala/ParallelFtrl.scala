/**
  * Created by Taehee on 2018. 5. 19..
  */


import breeze.linalg.SparseVector
import org.apache.spark.rdd.RDD


case class FtrlGlobalParam(
                      n: Int, i: Int,
                      alpha:Double, beta:Double, lambda:Double, lambda2:Double,
                      weight : SparseVector[Double], z : SparseVector[Double],
                      sigma : SparseVector[Double], cumGrad : SparseVector[Double],
                      cumGradSq : SparseVector[Double], nonZeroCoef : Int,
                      bufferSize:Int, buffer: Array[(Int, Double, Double, Array[Int])]
                    )




class Ftrl2 {

  var n = Int.MaxValue
  var i = 1
  var alpha:Double = 5.0
  var beta:Double = 1.0
  var lambda:Double = 0.0
  var lambda2:Double = 0.0
  var weight : SparseVector[Double] = SparseVector.zeros[Double](n)
  var z : SparseVector[Double] = SparseVector.zeros[Double](n)
  var sigma : SparseVector[Double] = SparseVector.zeros[Double](n)
  var cumGrad : SparseVector[Double] = SparseVector.zeros[Double](n)
  var cumGradSq : SparseVector[Double] = SparseVector.zeros[Double](n)
  var nonZeroCoef : Int = 0
  var bufferSize = 1000
  var buffer : Array[(Int, Double, Double, Array[Int])] =Array.empty
  var pCount : Map[Int, Double] = Map.empty
  var nCount : Map[Int, Double] = Map.empty

  var wMap : Map[Int, Double] = Map.empty
  var zMap : Map[Int, Double] = Map.empty
  var gMap : Map[Int, Double] = Map.empty


  def load(param:FtrlParam) = {
    this.n = param.n
    this.i = param.i
    this.alpha = param.alpha
    this.beta = param.beta
    this.lambda = param.lambda
    this.lambda2 = param.lambda2
    this.weight = param.weight
    this.z = param.z
    this.sigma = param.eta /** TO BE CORRECTED */
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
      this.sigma,
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

  def setW (w:Map[Int, Double]) ={
    this.wMap = w
    this.weight = FtrlRun.mapToSparseVector(wMap, n)
    this
  }
  def setP (p:Map[Int, Double]) ={
    this.pCount = p
    this
  }
  def setN (n:Map[Int, Double]) ={
    this.nCount = n
    this
  }

  def setPerCoordinateLearningRate (p:Map[Int, Double], n:Map[Int, Double]) = {
    this.pCount = p
    this.nCount = n
    this.cumGradSq = Ftrl2.cumGradSquareApprox(nCount, pCount)
    this.gMap = cumGradSq.activeIterator.toMap
    this
  }



  def update(data : (Int, SparseVector[Double])) = {

    val fit = fitStat(data)
    //val updatedParam = Ftrl2.ftrlUpdate(data,
    //  alpha, beta, lambda, lambda2, wMap, zMap, sigma, gMap)
    val updatedParam = Ftrl2.ftrlUpdateApprox(data,
      alpha, beta, lambda, lambda2, wMap, zMap, sigma, gMap, nCount, pCount)

    //val updatedParam = Ftrl2.ftrlUpdateApprox(
    //  data, alpha, beta, lambda, lambda2, weight, z, eta, cumGradSq)
    this.wMap = updatedParam._1
    this.gMap = updatedParam._2
    this.zMap = updatedParam._3
    this.nCount = updatedParam._4
    this.pCount = updatedParam._5

    this.weight = FtrlRun.mapToSparseVector(wMap, this.n)
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

object Ftrl2 {


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

  def sigmaUpdater(
                    oldSigma : SparseVector[Double],
                    //oldCumGradSq : SparseVector[Double],
                    //newCumGradSq : SparseVector[Double],
                    oldCumGradSq:Map[Int, Double],
                    newCumGradSq:Map[Int, Double],
                    alpha : Double,
                    data : SparseVector[Double]
                  ) ={

    val newSigma = oldSigma.copy
    data.activeIterator.foreach{x=>
      //val newN = if (newCumGradSq.contains(x._1)) newCumGradSq.valueAt(x._1) else 0
      //val oldN = if (oldCumGradSq.contains(x._1)) oldCumGradSq.valueAt(x._1) else 0
      val newN = oldCumGradSq.getOrElse(x._1, 0D)
      val oldN = newCumGradSq.getOrElse(x._1, 0D)
      newSigma.update(x._1, (math.sqrt(newN) - math.sqrt(oldN))/alpha)
      //newSigma += (x._1 -> (math.sqrt(newN) - math.sqrt(oldN))/alpha)
    }
    newSigma

  }

  def sigmaUpdater(
                    oldCumGradSq:Map[Int, Double],
                    newCumGradSq:Map[Int, Double],
                    alpha : Double
                  ) ={

    val index = oldCumGradSq.keySet ++ newCumGradSq.keySet
    var newSigma : Map[Int, Double] = Map.empty
    index.foreach{x=>
      //val newN = if (newCumGradSq.contains(x._1)) newCumGradSq.valueAt(x._1) else 0
      //val oldN = if (oldCumGradSq.contains(x._1)) oldCumGradSq.valueAt(x._1) else 0
      val newN = oldCumGradSq.getOrElse(x, 0D)
      val oldN = newCumGradSq.getOrElse(x, 0D)
      //newSigma.updated(x, (math.sqrt(newN) - math.sqrt(oldN))/alpha)
      newSigma += (x -> (math.sqrt(newN) - math.sqrt(oldN))/alpha)
    }
    FtrlRun.mapToSparseVector(newSigma, Int.MaxValue)

  }


  def zUpdater(
                //oldZ : SparseVector[Double],
                oldZ:Map[Int, Double],
                grad : SparseVector[Double],
                sigma : SparseVector[Double],
                //w : SparseVector[Double],
                w:Map[Int, Double],
                data : SparseVector[Double]
              ) ={

    val gMap = grad.activeIterator.toMap
    val sMap = sigma.activeIterator.toMap
    var newZ = oldZ
    //oldZ + grad - sigma *:* w
    data.activeIterator.foreach{x=>
      val z = oldZ.getOrElse(x._1, 0D)
      //newZ = newZ.updated(x._1, z + gMap.getOrElse(x._1, 0D) - sMap.getOrElse(x._1, 0D) * w.getOrElse(x._1, 0D) )
      newZ += (x._1-> (z + gMap.getOrElse(x._1, 0D) - sMap.getOrElse(x._1, 0D) * w.getOrElse(x._1, 0D) ))
    }
    newZ
  }

  def wUpdater(
                //oldW : SparseVector[Double],
                oldW:Map[Int, Double],
                //z:SparseVector[Double],
                z:Map[Int, Double],
                //cumGradSq:SparseVector[Double],
                cumGradSq:Map[Int, Double],
                alpha : Double, beta : Double,
                lambda:Double, lambda2 :Double,
                data:SparseVector[Double]
              ) = {


    //val ni = cumGradSq.activeIterator.toMap
    //val zMap = z.activeIterator.toMap
    var newW = oldW

    data.activeIterator.foreach{x=>

      //val ni = if (cumGradSq.isActive(x._1)) cumGradSq.valueAt(x._1) else 0
      val ni = cumGradSq.getOrElse(x._1, 0D)//if (cumGradSq.contains(x._1)) cumGradSq.valueAt(x._1) else 0
      val zi = z.getOrElse(x._1, 0D)//if (z.contains(x._1)) z.valueAt(x._1) else 0

      val discount = -((beta + math.sqrt(ni)) / alpha + lambda2)
      if (math.abs(zi) <= lambda) newW += (x._1 -> 0D)//newW = newW.updated(x._1, 0)
      else {
        //newW = newW.updated(x._1, if(zi >=0) (zi -lambda) / discount else (zi+lambda) / discount)
        newW += (x._1 -> (if(zi >=0) (zi -lambda) / discount else (zi+lambda) / discount))
        //if (zi >= 0) newZ.update(x._1, zi - lambda) else newZ.update(x._1, zi + lambda)
      }
    }

    //newW.compact()

    newW
    //oldW
  }

  def cumGradSqUpdater(
                        //oldCumGradSq:SparseVector[Double],
                        oldCumGradSq:Map[Int, Double],
                        grad:SparseVector[Double],
                        data:SparseVector[Double]
                      ) = {


    val gradMap = grad.activeIterator.toMap
    var newCumGradSq = oldCumGradSq
    data.activeIterator.foreach{x=>

      val gradi = gradMap.getOrElse(x._1, 0D)
      //if (grad.contains(x._1)) grad.valueAt(x._1) else 0
      val gradSq = gradi * gradi
      //newCumGradSq.update(x._1, gradSq)
      //
      newCumGradSq += (x._1 -> gradSq)
    }
    //newCumGradSq.compact()
    newCumGradSq
    //oldCumGradSq

  }

  def counter(count:Map[Int,Double], data:SparseVector[Double]) = {

    var newCount = count
    //val index = SparseVector.zeros[Double](Int.MaxValue)
    data.activeIterator.foreach{i=>
      val c = count.getOrElse(i._1, 0D)
      newCount += (i._1 -> (c + 1))
    }
    //println(count + index)
    newCount
  }

  def cumGradSquareApprox(
                           nCount : Map[Int,Double],
                           pCount : Map[Int,Double],
                           oldCumGradSq : Map[Int, Double],
                           data : SparseVector[Double]
                         ) = {

    var newCumGradSq = oldCumGradSq

    data.activeIterator.foreach{x=>
      val nom = nCount.getOrElse(x._1, 0D) * pCount.getOrElse(x._1, 0D)
      val denom = nCount.getOrElse(x._1, 0D) + pCount.getOrElse(x._1, 0D)
      newCumGradSq += (x._1 -> nom / denom)
    }
    newCumGradSq
  }

  def cumGradSquareApprox(
                           nCount : Map[Int,Double],
                           pCount : Map[Int,Double]
                         ) = {

    val newCumGradSq = FtrlRun.mapToSparseVector(nCount ++ pCount, Int.MaxValue)

    newCumGradSq.activeIterator.foreach{x=>
      val nom = nCount.getOrElse(x._1, 0D) * pCount.getOrElse(x._1, 0D)
      val denom = nCount.getOrElse(x._1, 0D) + pCount.getOrElse(x._1, 0D)
      newCumGradSq.update(x._1, nom / denom)
    }
    newCumGradSq
  }

  def ftrlUpdateApprox(

                       data : (Int, SparseVector[Double]),
                       alpha:Double,
                       beta:Double,
                       lambda:Double,
                       lambda2:Double,
                       oldWeight:Map[Int, Double],
                       oldZ:Map[Int, Double],
                       oldSigma : SparseVector[Double],
                       oldCumGradSq:Map[Int, Double],
                       nCount:Map[Int, Double],
                       pCount:Map[Int, Double]
                ) = {

    val weight = wUpdater(oldWeight, oldZ, oldCumGradSq, alpha, beta, lambda, lambda2, data._2)
    val grad = gradientLL(data._1, FtrlRun.mapToSparseVector(weight, Int.MaxValue), data._2); grad.compact()

    var newP = pCount
    var newN = nCount
    if (data._1 == 1) newP = counter(pCount, data._2) else newN = counter(nCount, data._2)

    val cumGradSq = cumGradSquareApprox(newN, newP, oldCumGradSq, data._2)
    val newSigma = sigmaUpdater(oldSigma, oldCumGradSq, cumGradSq, alpha, data._2)
    //val newEta = eta(cumGradSq, alpha, beta, lambda2, data._2); newEta.compact
    val newZ = zUpdater(oldZ, grad, newSigma, weight, data._2)

    //println(weight)
    //println(weight.keySet)
    (weight.filter(x=> x._2 !=0), cumGradSq.filter(x=> x._2 !=0),
      newZ.filter(x=> x._2 !=0), newP.filter(x=> x._2 !=0), newN.filter(x=> x._2 !=0))

  }

  def ftrlUpdate(
                       data : (Int, SparseVector[Double]),
                       alpha:Double,
                       beta:Double,
                       lambda:Double,
                       lambda2:Double,
                       //oldWeight : SparseVector[Double],
                       oldWeight:Map[Int, Double],
                       //oldZ : SparseVector[Double],
                       oldZ:Map[Int, Double],
                       oldSigma : SparseVector[Double],
                       //oldCumGradSq : SparseVector[Double]
                       oldCumGradSq:Map[Int, Double]
                     ) = {


    val weight = wUpdater(oldWeight, oldZ, oldCumGradSq, alpha, beta, lambda, lambda2, data._2)//; weight.compact()
    val grad = gradientLL(data._1, FtrlRun.mapToSparseVector(weight, Int.MaxValue), data._2); grad.compact()
    //val cumGrad = oldCumGrad + grad; cumGrad.compact()
    val cumGradSq = cumGradSqUpdater(oldCumGradSq, grad, data._2)//; cumGradSq.compact()
    val newSigma = if (oldSigma.activeSize > 0) {
      sigmaUpdater(oldSigma, oldCumGradSq, cumGradSq, alpha, data._2)
    } else sigmaUpdater(oldCumGradSq, cumGradSq, alpha)
    //val newEta = eta(cumGradSq, alpha, beta, lambda2, data._2); newEta.compact
    val newZ = zUpdater(oldZ, grad, newSigma, weight, data._2)//; newZ.compact()

    //println(newZ.size)
    (weight.filter(x=> x._2 !=0), cumGradSq.filter(x=> x._2 !=0), newZ.filter(x=> x._2 !=0))

  }
}

