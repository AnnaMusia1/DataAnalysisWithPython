import pandas as pd
other_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(other_path, header=None)
print("The last 10 rows of the dataframe")
print(df.tail(10))

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
           "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
           "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
           "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)
df.columns = headers
df.head(10)

# We need to replace the "?" symbol with NaN so the .dropna() can remove the missing values:
df1 = df.replace('?', "NaN")
# We  drop missing values along the column "price" as follows:
df=df1.dropna(subset=["price"], axis=0)
print(df.head(20))

# to check the headers:
print(df.columns)