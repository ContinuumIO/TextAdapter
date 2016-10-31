# !!! NOTE: YOU MUST HAVE STARTED A MONGO DEMON ON PORT 27017 BEFORE RUNNING THIS. See: mongod

import iopro
import pymongo

# open and populate a mongodb
hostname='localhost'
port=27017
db_name = 'MongoAdapter_tests'
colxn_name = 'test_floats'
nrecords = 5
try:
    mongo_conn = pymongo.MongoClient(hostname, port)
except:
   raise Exception("Could not connect to Mongo database. You must have a mongo demon running on port 27017 for this to work. Have you run '$ mongod'?")

mongo_db = mongo_conn[db_name]

col = mongo_db[colxn_name]
if col.count() > 0:
    col.remove()
for i in range(nrecords):
    record = {'f0': float(i * 5 + 0.1),
              'f1': float(i * 5 + 1 + 0.1),
              'f2': float(i * 5 + 2 + 0.1),
              'f3': float(i * 5 + 3 + 0.1),
              'f4': float(i * 5 + 4 + 0.1)}
    col.insert(record)

# test out iopro's MongoAdapter
adapter = iopro.MongoAdapter(hostname, port, db_name, colxn_name)

adapter[['f0', 'f1', 'f2', 'f3', 'f4']] # this is equivalent to set_field_names(['f0', 'f1', 'f2', 'f3', 'f4'])

print('get_field_names(): ' + repr(adapter.get_field_names()))
adapter.set_field_types({'f1':'i8'})
print("get_field_types() = " + repr(adapter.get_field_types()))
print("adapter[:] = \n" + repr(adapter[:]))
