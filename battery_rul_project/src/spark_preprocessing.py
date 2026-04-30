from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min, max, input_file_name, regexp_extract, expr
from pyspark.sql.functions import max as spark_max
from pyspark.sql.window import Window
from pyspark.sql.functions import lag
import os

# -----------------------------
# 0. SET PROJECT PATHS (IMPORTANT FIX)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_PATH = os.path.join(BASE_DIR, "processed_data")
SPARK_OUTPUT_PATH = os.path.join(PROCESSED_PATH, "spark_output")
MONGO_PATH = os.path.join(PROCESSED_PATH, "mongo_ready")
LIGHT_PATH = os.path.join(PROCESSED_PATH, "light_cleaned")

# -----------------------------
# 1. Start Spark
# -----------------------------
spark = SparkSession.builder \
    .appName("BatteryRUL") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# -----------------------------
# 2. LOAD FROM HDFS
# -----------------------------
hdfs_path = "hdfs://localhost:9000/battery_data/*/*.csv"

df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(hdfs_path)

print("✅ Total rows loaded from HDFS:", df.count())

# -----------------------------
# 3. ADD battery_id + battery_type
# -----------------------------
df = df.withColumn("file_path", input_file_name())

# Extract battery_id
df = df.withColumn(
    "battery_id",
    regexp_extract("file_path", r'([^/]+)\.csv', 1)
)

# Extract battery_type (HDFS path)
df = df.withColumn(
    "battery_type",
    regexp_extract("file_path", r"battery_data/([^/]+)/", 1)
)

# -----------------------------
# 4. SAFE TYPE CONVERSION
# -----------------------------
df = df.withColumn("time", expr("try_cast(time as double)")) \
       .withColumn("mode", expr("try_cast(mode as double)")) \
       .withColumn("voltage_load", expr("try_cast(voltage_load as double)")) \
       .withColumn("current_load", expr("try_cast(current_load as double)")) \
       .withColumn("temperature_battery", expr("try_cast(temperature_battery as double)"))

# -----------------------------
# 5. FILTER DISCHARGE DATA
# -----------------------------
df = df.filter(col("mode") == -1)

print("✅ After discharge filter:", df.count())

# -----------------------------
# 6. CLEAN DATA
# -----------------------------
df = df.dropna()

# -----------------------------
# ⭐ SAVE LIGHT DATASET
# -----------------------------
df_light = df.select(
    "battery_id",
    "battery_type",
    "time",
    "voltage_load",
    "current_load",
    "temperature_battery"
)

df_light.write.mode("overwrite").option("header", True).csv(LIGHT_PATH)

print("✅ Light dataset saved")

# -----------------------------
# 7. CREATE CYCLE COLUMN
# -----------------------------
df = df.withColumn("cycle", (col("time") / 1000).cast("int"))

# -----------------------------
# 8. AGGREGATION
# -----------------------------
agg = df.groupBy("battery_id", "battery_type", "cycle").agg(
    avg("voltage_load").alias("avg_voltage"),
    min("voltage_load").alias("min_voltage"),
    avg("current_load").alias("avg_current"),
    max("temperature_battery").alias("max_temp"),
    max("time").alias("t_end"),
    min("time").alias("t_start")
)

agg = agg.withColumn("duration", col("t_end") - col("t_start"))
agg = agg.drop("t_start", "t_end")

print("✅ Aggregated rows:", agg.count())

# -----------------------------
# 9. RUL CALCULATION
# -----------------------------
max_cycle_df = agg.groupBy("battery_id").agg(
    spark_max("cycle").alias("max_cycle")
)

agg = agg.join(max_cycle_df, on="battery_id")

agg = agg.withColumn("RUL", col("max_cycle") - col("cycle"))
agg = agg.withColumn("SoH", 1 - (col("cycle") / col("max_cycle")))

print("✅ RUL added")
print("✅ SoH added")

# -----------------------------
# 10. FEATURE ENGINEERING
# -----------------------------
window = Window.partitionBy("battery_id").orderBy("cycle")

agg = agg.withColumn("prev_voltage", lag("avg_voltage").over(window))
agg = agg.withColumn("voltage_drop", col("avg_voltage") - col("prev_voltage"))

agg = agg.withColumn("prev_temp", lag("max_temp").over(window))
agg = agg.withColumn("temp_change", col("max_temp") - col("prev_temp"))

agg = agg.withColumn("cycle_ratio", col("cycle") / col("max_cycle"))

# Remove null rows created by lag
agg = agg.dropna()

print("✅ Feature engineering complete")

# -----------------------------
# 11. FINAL OUTPUTS
# -----------------------------
agg.write.mode("overwrite").option("header", True).csv(SPARK_OUTPUT_PATH)
agg.write.mode("overwrite").json(MONGO_PATH)

print("🚀 FULL Spark pipeline complete!")
print("📂 Light dataset:", LIGHT_PATH)
print("📂 Final dataset:", SPARK_OUTPUT_PATH)
print("📂 MongoDB JSON:", MONGO_PATH)
