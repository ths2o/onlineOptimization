/**
  * Created by Taehee on 2018. 5. 19..
  */


import breeze.linalg.SparseVector


object LogisticRandomSampling {



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

    /*
    val aa = (1 to 8000).map(x=> makeCoef(5, 100)) union (1 to 2000).map(x=> makeCoef(0, 100000))

    val bb = util.Random.shuffle(aa).toArray.
      map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
      flatten

    val data = bb

    val ss = data.map{x=>
      val label = x._1.toString
      val feature = x._2.array.toMap.map(k=> k._1.toString + ":"+ k._2.toString).mkString(" ")
      //"echo " + "\""+ label + " " + feature + "\"" + "| nc 127.0.0.1 9999"
      label + " " + feature
    }
*/

    def generateOne () = {

      val tt = rBernoulli(1, 0.8).map{x=>
        if (x == 1) makeCoef(5, 100) else makeCoef(0, 100000)
      }

      val bb = tt.toArray.
        map(x=> nLogisticSample(1, mapToSparseVector(x, Int.MaxValue))).
        flatten

      val data = bb

      val ss = data.map{x=>
        val label = x._1.toString
        val feature = x._2.array.toMap.map(k=> k._1.toString + ":"+ k._2.toString).mkString(" ")
        //"echo " + "\""+ label + " " + feature + "\"" + "| nc 127.0.0.1 9999"
        label + " " + feature
      }
      Thread.sleep(10)
      ss
    }


    import java.io.FileOutputStream
    import java.io.PrintStream


    //writer.flush()
    (0 to 10000).foreach{x=>

      val writer = new PrintStream(new FileOutputStream("/Users/Taehee/Documents/project/kafka_2.11-1.1.0/test.txt", true))
      val sample = generateOne()(0)
      println(sample)
      writer.append(sample + "\n")
      writer.close()

    }



  }
}
