import iopro

adapter = iopro.CSVTextAdapter('../tests/data/ints', delimiter=',', field_names=False)

# Set dtype for each field in record
adapter.set_field_types({0:'u4', 1:'u8', 2:'f4', 3:'f8', 4:'S10'})

# Read all records
print(adapter[:])

# Read first ten records
print(adapter[0:10])

# Change dtype; retrieve only 1st and 5th field
adapter.set_field_types({0:'u4', 4:'u4'})

# Read every other record
print(adapter[::2])

