import iopro

adapter = iopro.CSVTextAdapter('../tests/data/missingvalues', delimiter=',', field_names=False)

# Set dtype for each field in record
adapter.set_field_types({0:'u4', 1:'u4', 2:'u4', 3:'u4', 4:'u4'})

# Define list of strings for each field that represent missing values
adapter.set_missing_values({0:['NA', 'NaN'], 4:['xx','inf']})

# Set fill value for missing values in each field
adapter.set_fill_values({0:99, 4:999})

# Read all records
print(adapter[:])


