import iopro
import unittest
import warnings
import numpy


try:
    import pymongo
    pymongo_installed = True
except ImportError:
    pymongo_installed = False

nrecords = 50 # number of records held by each table in the tests

class TestMongoAdapter(unittest.TestCase):

    hostname = 'localhost'
    port = 27017
    db_name = 'MongoAdapter_tests'


    def test_uints(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_uints')
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[i].item(), (i*5, i*5+1, i*5+2, i*5+3, i*5+4))


    def test_ints(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_ints')
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('i8,i8,i8,i8,i8'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[nrecords-1-i].item(), (i*-5, i*-5+1, i*-5+2, i*-5+3, i*-5+4))


    def test_floats(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_floats')
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('f8,f8,f8,f8,f8'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[i].item(), (i*5 + 0.1 , i*5 + 0.1 +1, i*5 + 0.1 +2, i*5 + 0.1 +3, i*5 + 0.1 +4))

 
    def test_behavior_of_floats_that_hold_uints(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_behavior_of_floats_that_hold_uints')
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[i].item(), (i*5, i*5+1, i*5+2, i*5+3, i*5+4))

    
    def test_type_inference(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_type_inference')
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('f8,O,u8,u8,u8'))
        sorted = numpy.sort(array)
        for i in range(nrecords - 1):
            self.assertEqual(sorted[i].item(), (float(i*5), str(i*5+1), i*5+2, i*5+3, i*5+4))
        self.assertEqual(sorted[-1].item(), (float(4995.5), '4996', 4997, 4998, 4999))


    def test_field_subset(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_uints')

        array = adapter['f0'][:]
        array.sort()
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype([('f0', 'u8')]))
        for i in range(nrecords):
            self.assertEqual(array[i].item(), (i*5,))

        array = adapter[['f0', 'f4']][:]
        array.sort()
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype([('f0', 'u8'),('f4', 'u8')]))
        for i in range(nrecords):
            self.assertEqual(array[i].item(), (i*5, i*5+4))

        array = adapter[('f0', 'f1', 'f2', 'f3', 'f4')][:]
        array.sort()
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        for i in range(nrecords):
            self.assertEqual(array[i].item(), (i*5, i*5+1, i*5+2, i*5+3, i*5+4))

    def test_array_slicing(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_uints')
        control = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]

        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][0:nrecords]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        for i in range(nrecords):
            self.assertEqual(array[i].item(), control[i].item())
       
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][-1]
        self.assertEqual(array.size, 1)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        self.assertEqual(array[0].item(), control[-1].item())
 
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][::2]
        self.assertEqual(array.size, nrecords // 2)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        for i in range(0, nrecords // 2, 2):
            self.assertEqual(array[i].item(), control[i*2].item())

    def test_further_slicing(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_ints')
        
        array_list_all = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        self.assertEqual(array_list_all.size, nrecords)

        array_str_1011 = adapter['f0'][10:11]
        self.assertEqual(array_str_1011.size, 1)

        array_str_int = adapter['f0'][10]
        self.assertEqual(array_str_int.size, 1)
        self.assertEqual(array_str_int, array_str_1011)

        array_list_10 = adapter[['f0']][10]
        self.assertEqual(array_str_int, array_list_10)

        array_list_515 = adapter[['f0']][5:15]
        self.assertEqual(array_list_515.size, 10)
        self.assertEqual(array_list_515[5:6], array_str_int)
        for i in range(5):
            self.assertEqual(array_list_515[5 + i][0], array_list_all[10 + i][0])
        self.assertTrue(numpy.allclose(array_list_515['f0'][5:10], \
                                       array_list_all['f0'][10:15]))


    @unittest.expectedFailure # getting item from adapter without specifying fields fails
    def test_slice_no_fields(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_ints')
        array = adapter[:]
        self.assertEqual(array.size, nrecords)

    @unittest.expectedFailure # Currently, asking for records as a list or tuple of ints will fail.
    def test_int_list(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_ints')
        array = adapter['f0'][[0,1,2]]
        self.assertEqual(array.size, 3)


    def test_set_field_types(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port,
            TestMongoAdapter.db_name, 'test_ints')
        adapter[['f0', 'f1', 'f2', 'f3', 'f4']]
        adapter.set_field_types({'f0':'f8', 'f4':'O'})
        array = adapter[:]        
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('f8,i8,i8,i8,O'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[nrecords-1-i].item(), (float(i*-5), i*-5+1, i*-5+2, i*-5+3, str(i*-5+4)))


    def test_cast_to_uint_from_int(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port,
            TestMongoAdapter.db_name, 'test_ints')
        arr1 = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        adapter.set_field_types({'f0':'u8', 'f1':'u8','f2':'u8','f3':'u8','f4':'u8'})
        self.assertTrue(adapter.is_field_type_set('f1'))
        for i in range(5):
            self.assertTrue(adapter.is_field_type_set(i))
        array = adapter[:] 
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[-i].item(), ((i*-5+0) & 0xffffffffffffffff, \
                                                (i*-5+1) & 0xffffffffffffffff, \
                                                (i*-5+2) & 0xffffffffffffffff, \
                                                (i*-5+3) & 0xffffffffffffffff, \
                                                (i*-5+4) & 0xffffffffffffffff))


    def test_cast_to_int_from_float(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, 'test_floats')
        adapter[['f0', 'f1', 'f2', 'f3', 'f4']]
        adapter.set_field_types({'f0':'i8', 'f1':'i8','f2':'i8','f3':'i8','f4':'i8'})
        array = adapter[:]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('i8,i8,i8,i8,i8'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[i].item(), (i*5  , i*5  +1, i*5  +2, i*5  +3, i*5  +4))

    @unittest.skip('test_missing_fill_values')
    def test_missing_fill_values(self):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port,
            TestMongoAdapter.db_name, 'test_missing_ints', infer_types=False)

        # Force f1 type to integer
        adapter.set_field_types({'f1':'u8'})

        adapter.set_missing_values({0:['NA']})
        adapter.set_fill_values({0:99, 1:999, 4:9999}, loose=True)
        
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))

        # Records 0,5,10... contains 'NA' and 'NaN' values in f0
        # that should convert to 99.
        # Records 1,6,11... contains values in f1 that can't be converted to int;
        # these should convert to 999.
        # Records 4,9,14... contains missing values in f4
        # that should convert to 9999.
        for i in range(nrecords):
            if i % 5 == 0:
                self.assertEqual(array[i], (99, i*5+1, i*5+2, i*5+3, i*5+4))
            elif i % 5 == 1:
                self.assertEqual(array[i], (i*5, 999, i*5+2, i*5+3, i*5+4))
            elif i % 5 == 4:
                self.assertEqual(array[i], (i*5, i*5+1, i*5+2, i*5+3, 9999))
            else:
                self.assertEqual(array[i], (i*5, i*5+1, i*5+2, i*5+3, i*5+4))
    
    def run_adapter_vs_pymongo(self, col_name, cast=False):
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname, TestMongoAdapter.port, TestMongoAdapter.db_name, col_name)
        array = adapter[['f0', 'f1', 'f2', 'f3', 'f4']][:]
        try:
            mongo_conn = pymongo.MongoClient(TestMongoAdapter.hostname, TestMongoAdapter.port)
        except:
            warnings.warn('Could not run Mongo tests: could not connect to Mongo database.')
            self.assertTrue(True)
            return    
        mongo_db = mongo_conn['MongoAdapter_tests']
        col = mongo_db[col_name]        
        i = 0
        for rec in col.find():
            self.check_field('f0', array, i, rec, cast)
            self.check_field('f1', array, i, rec, cast)
            self.check_field('f2', array, i, rec, cast)
            self.check_field('f3', array, i, rec, cast)
            self.check_field('f4', array, i, rec, cast)
            i += 1

    def check_field(self, field_name, array, index, record, cast = False):
            adapter_val = array[field_name][index]
            pymongo_val = record[field_name]
            if cast:
                self.assertEqual(adapter_val, type(adapter_val)(pymongo_val))
            else:
                self.assertEqual(adapter_val, pymongo_val)
    
    def test_adapter_vs_pymongo_ints(self):
        self.run_adapter_vs_pymongo('test_ints')

    def test_adapter_vs_pymongo_floats(self):
        self.run_adapter_vs_pymongo('test_floats')

    def test_adapter_vs_pymongo_strings(self):
        self.run_adapter_vs_pymongo('test_strings')
    
    @unittest.expectedFailure
    # CSC: MongoDB supports different records containing diff types for same field. Numpy doesn't
    def test_adapter_vs_pymongo_type_inference(self):
        self.run_adapter_vs_pymongo('test_type_inference')
    # CSC: If we cast each field and compare, we're ok
    def test_adapter_vs_pymongo_type_inference_cast(self):
        self.run_adapter_vs_pymongo('test_type_inference', cast = True)   

    def test_slice_overflow(self):
        """
        Test using a value for slice end that is larger than max uint32 size.
        Expected behavior is to round slice end down to max uint32 value
        (effectively reading as many records as possible).
        """
        adapter = iopro.MongoAdapter(TestMongoAdapter.hostname,
                                     TestMongoAdapter.port,
                                     TestMongoAdapter.db_name,
                                     'test_uints')
        adapter.set_field_names(['f0', 'f1', 'f2', 'f3', 'f4'])
        array = adapter[0:999999999999999999]
        self.assertEqual(array.size, nrecords)
        self.assertEqual(array.dtype, numpy.dtype('u8,u8,u8,u8,u8'))
        sorted = numpy.sort(array)
        for i in range(nrecords):
            self.assertEqual(sorted[i].item(), (i*5, i*5+1, i*5+2, i*5+3, i*5+4))


def run(verbosity=2, hostname='localhost', port=27017):

    TestMongoAdapter.hostname = hostname
    TestMongoAdapter.port = port

    if not pymongo_installed:
        warnings.warn('Could not run Mongo tests: pymongo is not installed.')
        return

    try:
        mongo_conn = pymongo.MongoClient(TestMongoAdapter.hostname, TestMongoAdapter.port)
    except:
        warnings.warn('Could not run Mongo tests: could not connect to Mongo database.')
        return

    mongo_db = mongo_conn['MongoAdapter_tests']

    col = mongo_db['test_uints']
    if col.count() > 0:
        col.remove()
    for i in range(nrecords):
        record = {'f0': i * 5,
                  'f1': i * 5 + 1,
                  'f2': i * 5 + 2,
                  'f3': i * 5 + 3,
                  'f4': i * 5 + 4}
        col.insert(record)

    col = mongo_db['test_ints']
    if col.count() > 0:
        col.remove()
    for i in range(nrecords):
        record = {'f0': i * -5,
                  'f1': i * -5 + 1,
                  'f2': i * -5 + 2,
                  'f3': i * -5 + 3,
                  'f4': i * -5 + 4}
        col.insert(record)

    col = mongo_db['test_floats']
    if col.count() > 0:
        col.remove()
    for i in range(nrecords):
        record = {'f0': float(i * 5 + 0.1),
                  'f1': float(i * 5 + 1 + 0.1),
                  'f2': float(i * 5 + 2 + 0.1),
                  'f3': float(i * 5 + 3 + 0.1),
                  'f4': float(i * 5 + 4 + 0.1)}
        col.insert(record)

    col = mongo_db['test_behavior_of_floats_that_hold_uints']
    if col.count() > 0:
        col.remove()
    for i in range(nrecords):
        record = {'f0': float(i * 5),
                  'f1': float(i * 5 + 1),
                  'f2': float(i * 5 + 2),
                  'f3': float(i * 5 + 3),
                  'f4': float(i * 5 + 4)}
        col.insert(record)

    col = mongo_db['test_strings']
    if col.count() > 0:
        col.remove()
    for i in range(nrecords):
        record = {'f0': str(i * 5),
                  'f1': str(i * 5 + 1),
                  'f2': str(i * 5 + 2),
                  'f3': str(i * 5 + 3),
                  'f4': str(i * 5 + 4)}
        col.insert(record)

    col = mongo_db['test_missing_ints']
    if col.count() > 0:
        col.remove()
    for i in range(nrecords):
        if i % 5 == 0:
            record = {'f0': 'NA',
                      'f1': i * 5 + 1,
                      'f2': i * 5 + 2,
                      'f3': i * 5 + 3,
                      'f4': i * 5 + 4}
        elif i % 5 == 1:
            record = {'f0': i * 5,
                      'f1': 'xxx',
                      'f2': i * 5 + 2,
                      'f3': i * 5 + 3,
                      'f4': i * 5 + 4}
        elif i % 5 == 4:
            record = {'f0': i * 5,
                      'f1': i * 5 + 1,
                      'f2': i * 5 + 2,
                      'f3': i * 5 + 3}
        else:
            record = {'f0': i * 5,
                      'f1': i * 5 + 1,
                      'f2': i * 5 + 2,
                      'f3': i * 5 + 3,
                      'f4': i * 5 + 4}
        col.insert(record)

    col = mongo_db['test_type_inference']
    if col.count() > 0:
        col.remove()
    for i in range(nrecords - 1):
        record = {'f0': i * 5,
                  'f1': i * 5 + 1,
                  'f2': i * 5 + 2,
                  'f3': i * 5 + 3,
                  'f4': i * 5 + 4}
        col.insert(record)
    col.insert({'f0':4995.5, 'f1':'4996', 'f2':4997, 'f3':4998, 'f4':4999})
       
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMongoAdapter)
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)


if __name__ == '__main__':
    run()
