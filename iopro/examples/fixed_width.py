import iopro

adapter = iopro.FixedWidthTextAdapter('../tests/data/fixedwidths', [2,3,4,5,6])

# Set dtype for each field in record
adapter.set_field_types(dict(zip(range(5), ['u4']*5)))

# Read all records
print(adapter[:])


