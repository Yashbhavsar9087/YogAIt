import time, datetime
import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["yogait"]
mycol = mydb["customers"]

print(time.time())
print(datetime.date.today())