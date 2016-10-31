import iopro

adapter = iopro.CSVTextAdapter('../tests/data/ints', delimiter=',', field_names=False)

# Set dtype for each field in record
adapter.set_field_types({0:'u4', 1:'u8', 2:'f4', 3:'f8', 4:'S10'})

# Override default converter for first field
adapter.set_converter(0, lambda x: int(x)*2)

# Read first 10 records
print(adapter[:10])

