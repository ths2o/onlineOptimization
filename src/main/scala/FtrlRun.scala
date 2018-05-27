/**
  * Created by Taehee on 2018. 5. 19..
  */


import breeze.linalg.SparseVector


object FtrlRun {



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


  def sigmoid(a:Double) = {
    1 / (1 + math.exp(-a))
  }




  def main(args: Array[String]): Unit = {


    val coef = Map(0 -> 1.0, 1-> 2.0, 2->3.0, 3-> 4.0, 4-> 5.0, 5-> 0D, 9->0D)

    val coef2 = Map(0 -> 1.0, 1-> 2.0, 2->3.0, 3-> 4.0, 4-> 5.0, 6-> 15D, 8->7D)

    println(mapToSparseVector(coef, Int.MaxValue))//.index.max)
    //val sampledData = nLogisticSample(500000, mapToSparseVector(coef, Int.MaxValue)).toArray
    //val sampledData2 = nLogisticSample(500000, mapToSparseVector(coef2, Int.MaxValue)).toArray
    //val data =sampledData.union(sampledData2)
    //val data = nLogisticSample(1000000, mapToSparseVector(makeCoef(), Int.MaxValue)).toArray
    //println(data.mkString("\n"))
/*
    val data1 = (1 to 80000).
      map(x=> makeCoef(5, 100)).
      map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
      flatten.toArray


    val data2 = (1 to 20000).
      map(x=> makeCoef(0, 10000)).
      map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
      flatten.toArray
    */
    val aa = (1 to 80000).map(x=> makeCoef(5, 100)) union (1 to 20000).map(x=> makeCoef(0, 100000))

    val bb = util.Random.shuffle(aa).toArray.
      map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
      flatten

    val data = bb
    //println(data.mkString("\n"))

    val ss = data.map{x=>
      val label = x._1.toString
      val feature = x._2.array.toMap.map(k=> k._1.toString + ":"+ k._2.toString).mkString(" ")
      //"echo " + "\""+ label + " " + feature + "\"" + "| nc 127.0.0.1 9999"
      label + " " + feature
    }

    println(ss.mkString("\n"))
    //val initialWeight = SparseVector.zeros[Double](coef.keys.max + 1)
    //gradientDescent(data, 10, 1, 2, initialWeight, 1, 0.00001)


    val hyperParam = Array(
      (1, 1, 0, 0)//,  (1, 1, 3, 0)
    )

    //val opt1 = new Ftrl().setAlpha(5).setBeta(1).setL1(0.5).setL2(1)
    //val opt2 = new Ftrl().setAlpha(5).setBeta(1)
    //val opt2 = new Ogd()
    //val opt = Array(opt1, opt2)

    val opt = hyperParam.map(h=> new Ftrl().setAlpha(h._1).setBeta(h._2).setL1(h._3).setL2(h._4))


    val t1 = System.currentTimeMillis()
    var i = 1
    var correct = Array.fill(hyperParam.size)(0)
    data.foreach{x=>

      val pred = opt.map(o=> o.predictLabel(x._2, 0.5))

      val prob = opt.map(o=> o.predictProb(x._2))
      val aa = opt.map(o=> o.update(x._1, x._2))

      val ss = pred.map{c=>
        if (c == x._1) 1 else 0
      }

      correct = correct.zip(ss).map(k => k._1 + k._2)

      val gg = correct.map(k=> k.toDouble / 1000).map(k=> k - (k % 0.0001))


      if (i % 1000 == 0) {
        val summary = opt.map(o=> o.bufferSummary(0.5))
        println(summary.mkString(","), i)
        correct = Array.fill(hyperParam.size)(0)
      }
      //if (i % 1000 == 0) println(opt2.i, opt2.n, opt2.weight, i)
      i += 1
    }

    val t2 = System.currentTimeMillis()
    println(t2-t1)



  }
}
