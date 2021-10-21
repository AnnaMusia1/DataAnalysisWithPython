# from dbmodule import connect
from sqlite3 import connect

# create connection object
# .connect() function returns connection object
# connection = connect("database", "username", "password")
connection = connect("data.db")

# create a cursor object
cursor = connection.cursor()

# run queries
cursor.execute("select * from Task")
results = cursor.fetchall()
print(results)

# free resources
cursor.close()
connection.close()


