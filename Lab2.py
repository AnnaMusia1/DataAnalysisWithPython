import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

other_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
           "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
           "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
           "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(other_path, names=headers)

# see how the data set looks like
print(df.head())

# Identifying missing data:
# replace "?" with empty input np.nan
df.replace("?", np.nan, inplace=True)
print(df.head())

# Evaluating the missing Data
# True stands for missing value, while "False" stands for not missing value
missing_data = df.isnull()
print(missing_data.head(5))

# Count missing values in each column
# we use .value_counts() method to calculate "True" values
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print()

# Based on the summary above, each column has 205 rows of data, seven columns containing missing data:
#
# "normalized-losses": 41 missing data
# "num-of-doors": 2 missing data
# "bore": 4 missing data
# "stroke" : 4 missing data
# "horsepower": 2 missing data
# "peak-rpm": 2 missing data
# "price": 4 missing data

# Deal with missing data
# How to deal with missing data?
# drop data
# a. drop the whole row
# b. drop the whole column
# replace data
# a. replace it by mean
# b. replace it by frequency
# c. replace it based on other functions
# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely. We have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. We will apply each method to many different columns:
#
# Replace by mean:
#
# "normalized-losses": 41 missing data, replace them with mean
#     "stroke": 4 missing data, replace them with mean
#     "bore": 4 missing data, replace them with mean
#     "horsepower": 2 missing data, replace them with mean
#     "peak-rpm": 2 missing data, replace them with mean
#     Replace by frequency:
#
# "num-of-doors": 2 missing data, replace them with "four".
#     Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur
# Drop the whole row:
#
# "price": 4 missing data, simply delete the whole row
# Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us

# Calculate the average of the column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
# Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

# Calculate the mean value for 'bore' column
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
# Replace NaN by mean value
df["bore"].replace(np.nan, avg_bore, inplace=True)

# Calculate the mean value for 'stroke' column
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke", avg_stroke)
# Replace NaN by mean value
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

# Calculate the mean value for the 'horsepower' column:
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
# Replace "NaN" by mean value:
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

# Calculate the mean value for 'peak-rpm' column:
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
# Replace NaN by mean value:
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

# To see which values are present in a particular column, we can use the ".value_counts()" method:
print(df['num-of-doors'].value_counts())

# We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate for us the most common type automatically:
print(df['num-of-doors'].value_counts().idxmax())

# replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# Finally, let's drop all rows that do not have price data:
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

print(df.head())


# CORRECT DATA FORMAT:
# Lets list the data types for each column
print(df.dtypes)

# Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print(df.dtypes)

# DATA STANDARDIZATION

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-mpg'] = 235/df["city-mpg"]
df.rename(columns={"city-mpg": "city- L/100km"}, inplace=True)

# check your transformed data
print(df.head())



# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={"highway-mpg":'highway-L/100km'}, inplace=True)

# check your transformed data
print(df.head())


# DATA NORMALIZATION
# To demonstrate normalization, let's say we want to scale the columns "length", "width" and "height"
# Target:would like to Normalize those variables so their value ranges from 0 to 1.
# Approach: replace original value by (original value)/(maximum value)

# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df["height"] = df["height"]/df["height"].max()

# show the scaled columns
print(df[["length","width","height"]].head())


# DATA BINNING
df["horsepower"]=df["horsepower"].astype(int, copy=True)

# Lets plot the histogram of horspower, to see what the distribution of horsepower looks like.
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.
# Since we want to include the minimum value of horsepower we want to set start_value=min(df["horsepower"]).
# Since we want to include the maximum value of horsepower we want to set end_value=max(df["horsepower"]).
# Since we are building 3 bins of equal length, there should be 4 dividers, so numbers_generated=4.
# We build a bin array, with a minimum value to a maximum value, with bandwidth calculated above.
# The bins will be values used to determine when one bin ends and another begins.

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
# We set group names:
group_names = ['Low', 'Medium', 'High']
# We apply the function "cut" the determine what each value of "df['horsepower']" belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))

# Lets see the number of vehicles in each bin.
print(df["horsepower-binned"].value_counts())

#Lets plot the distribution of each bin.
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# Bins visualization
# Normally, a histogram is used to visualize the distribution of bins we created above.

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# INDICATOR VARIABLE / DUMMY VARIABLE

# We will use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type.
print(df.columns)

# get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())

# change column names for clarity
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
print(dummy_variable_1.head())

# In the dataframe, column fuel-type has a value for 'gas' and 'diesel'as 0s and 1s now.
# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

print(df.head())

# create indicator variable to the column of "aspiration"
dummy_variable_1 = pd.get_dummies(df["aspiration"])

# Merge the new dataframe to the original dataframe then drop the column 'aspiration'
df = pd.concat([df, dummy_variable_1], axis=1)

# Save the new csv
path = "C:/Users/a.musial/IdeaProjects/DataAnalysisWithPython/clean_df.csv"
df.to_csv(path)
