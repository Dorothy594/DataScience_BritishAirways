import pandas as pd

df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
df.head()

df.info()

df["flight_day"].unique()

mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df["flight_day"] = df["flight_day"].map(mapping)

df["flight_day"].unique()

df.describe()

