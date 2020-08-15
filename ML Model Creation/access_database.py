import pymongo

conn = 'mongodb+srv://Client1:NecILf0FiqgirCeQ@classificationdatasetcl.qs98o.mongodb.net/Weather?retryWrites=true&w=majority'
client = pymongo.MongoClient(conn)
db = client.Weather
collection = db["classification"]

# print(collection.findOne())

print(
    collection.find({
        'City':'Denver',
        # 'Date and Time':''
    })
)
