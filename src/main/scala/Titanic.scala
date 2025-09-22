import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object Titanic {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
          .appName("Titanic Analysis")
          .master("local[*]")
          .getOrCreate()

        val df = spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(getClass.getResource("/train.csv").getPath)

        println("overview of the first 5 rows of data")
        df.show(5)


        println("Q1. What is the average ticket fare for each Ticket class?")
        df.groupBy("Pclass")
          .agg(round(avg("Fare"), 2).alias("Average fare"))
          .show()

        println("Q2. What is the survival percentage for each Ticket class? Which class has the highest survival rate?")
        df.groupBy("Pclass")
          .agg((round(sum("Survived") / count("Survived") * 100, 2)).alias("Survival Rate (%)"))
          .orderBy(col("Survival Rate (%)").desc)
          .show()

        println("Q3. find the number of passengers who could possibly be Rose?")
        val rose = df.filter(
            col("Age") === 17 &&
              col("Sex") === "female" &&
              col("Pclass") === 1 &&
              col("Parch") === 1
        )
        println(s"Number of possible candidates: ${rose.count()}")

        println("Q4. Find the number of passengers who could possibly be Jack?")
        val jack = df.filter(
            (col("Age") === 19 || col("Age") === 20) &&
              col("Sex") === "male" &&
              col("Pclass") === 3 &&
              col("SibSp") === 0 &&
              col("Parch") === 0
        )
        println(s"Number of possible candidates: ${jack.count()}")

        val maxAge = df.agg(max("Age")).first().getDouble(0).toInt

        val dfWithAgeGroup = df.withColumn(
            "AgeGroup",
            when(col("Age").isNotNull,
                when(col("Age") < 1, "1-10")
                  .otherwise(
                      concat(
                          ((ceil(col("Age") / 10) * 10 - 9).cast("int")),
                          lit("-"),
                          (ceil(col("Age") / 10) * 10).cast("int")
                      )
                  )
            ).otherwise("Unknown")
        )
        dfWithAgeGroup.show(5)

        dfWithAgeGroup.groupBy("AgeGroup")
          .agg(
              count("*").alias("Count"),
              round(avg("Fare"), 2).alias("Average Fare")
          )
          .orderBy("AgeGroup")
          .show(false)

        dfWithAgeGroup.groupBy("AgeGroup")
          .agg(
              count("*").alias("Count"),
              round(avg("Survived") * 100, 2).alias("Survival Rate (%)")
          )
          .orderBy(col("Survival Rate (%)").desc)
          .show(false)

    }
}