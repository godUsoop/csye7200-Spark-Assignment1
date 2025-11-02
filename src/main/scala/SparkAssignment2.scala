import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object SparkAssignment2 {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
            .appName("Titanic Assignment2")
            .master("local[*]")
            .getOrCreate()

        val train = spark.read.option("header", "true").option("inferSchema", "true").csv(getClass.getResource("/train.csv").getPath)
        val test = spark.read.option("header", "true").option("inferSchema", "true").csv(getClass.getResource("/test.csv").getPath)

        println(s"Train count = ${train.count()}, Test count = ${test.count()}")
        train.printSchema()

        // Data Analysis and Exploration
        train.describe("Age", "Fare").show()

        train.groupBy("Survived").count().show()
        train.groupBy("Sex").agg(avg("Survived").alias("SurvivalRate")).show()
        train.groupBy("Pclass").agg(avg("Survived").alias("SurvivalRate")).show()

        println("Most frequent value for each column in train dataset:")
        train.columns.foreach { c =>
            val modeValue = train.groupBy(c).count().orderBy(desc("count")).limit(1).collect()(0)(0)
            println(s"$c: $modeValue")
        }

        println("Missing value counts on train dataset:")
        train.columns.foreach { c =>
            val missing = train.filter(col(c).isNull || col(c) === "").count()
            println(s"$c: $missing missing")
        }

        println("Missing value counts on test dataset:")
        test.columns.foreach { c =>
            val missing = test.filter(col(c).isNull || col(c) === "").count()
            println(s"$c: $missing missing")
        }

        // Feature Engineering
        val trainFe = train
            .na.fill(Map("Age" -> 30, "Embarked" -> "S"))
            .withColumn("SexIndex", when(col("Sex") === "male", 1).otherwise(0))
            .withColumn("FamilySize", col("SibSp") + col("Parch") + lit(1))
            .withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))

        val testFe = test
            .na.fill(Map("Age" -> 30, "Fare" -> train.select(avg("Fare")).first().getDouble(0)))
            .withColumn("SexIndex", when(col("Sex") === "male", 1).otherwise(0))
            .withColumn("FamilySize", col("SibSp") + col("Parch") + lit(1))
            .withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))

        val featureCols = Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone")


        val assembler = new VectorAssembler()
            .setInputCols(featureCols)
            .setOutputCol("rawFeatures")

        val assembledTrain = assembler.transform(trainFe)
        val assembledTest = assembler.transform(testFe)

        // Standardized
        val scaler = new StandardScaler()
            .setInputCol("rawFeatures")
            .setOutputCol("features")
            .setWithMean(true)
            .setWithStd(true)

        val scalerModel = scaler.fit(assembledTrain)
        val scaledTrain = scalerModel.transform(assembledTrain)
        val scaledTest = scalerModel.transform(assembledTest)

        //  training model
        val lr = new LogisticRegression()
            .setMaxIter(200)
            .setRegParam(0.01)
            .setElasticNetParam(0.8)

        val trainLabeled = scaledTrain.withColumnRenamed("Survived", "label")

        val model = lr.fit(trainLabeled.select("features", "label"))

        // prediction and evaluation
        val predictions = model.transform(trainLabeled)
        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setMetricName("accuracy")

        val accuracy = evaluator.evaluate(predictions)
        println(f"Training Accuracy: ${accuracy * 100}%.2f%%")

        val testPredictions = model.transform(scaledTest)

        // show the first 10 result（1 = Survived, 0 = Dead）
        println("Predictions on test.csv:")
        testPredictions
            .select("PassengerId", "prediction")
            .withColumn("prediction", col("prediction").cast("Int"))
            .orderBy("PassengerId")
            .show(10, truncate = false)
    }
}