import iopro

adapter = iopro.CSVTextAdapter('../tests/data/ints.gz', delimiter=',', compression='gzip', field_names=False)

# Set dtype for each field in record
adapter.set_field_types({0:'u4', 1:'u8', 2:'f4', 3:'f8', 4:'S10'})

print('\n\n!!! INVESTIGATE !!!\n\n')

# adapter.size is unknown at this point...
print('Before we read any records, try adapter.size...')
try:
    sz = adapter.size
except AttributeError as err:
    print('AttributeError:', err)

# Read first record
print('Read first record\n', adapter[0])

# But now adapter.size IS known!
print('After we read a record...')
print('adapter.size', adapter.size)

# Read last record
print('\n\nNow we attempt to read the LAST record...')
print('adapter[-1] == adapter[0]?!? == ', adapter[-1])
print('adapter[1], as should be == ', adapter[1])

print('After we read ALL records...')
records = adapter[:]
print('adapter[-1] == ', adapter[-1])

print('\n\nFollowing code seems outdated. Remove it?') 
try:
    # build index of records and save index to file
    indexArray, gzipIndexArray = adapter.create_index()
    # load index from file
    adapter.set_index(indexArray, gzipIndexArray)
except TypeError as err:
    raise TypeError(err)



