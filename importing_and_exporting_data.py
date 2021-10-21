import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# our data has no headers, so we need to specify it
df = pd.read_csv(url, header=None)

# print(df)
print(df.head(3))

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels",
           "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-power", "num-of-cylinders",
           "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
           "highway-mpg", "price"]
df.columns = headers
print(df.head(3))

path = "C:/Users/a.musial/IdeaProjects/DataAnalysisWithPython/automobile.csv"
df.to_csv(path)

print(df.dtypes)

print(df.describe(include="all"))

print(df.info())
