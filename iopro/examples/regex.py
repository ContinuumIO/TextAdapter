import iopro

adapter = iopro.RegexTextAdapter('../tests/data/ints', '([0-9]*),([0-9]*),([0-9]*),([0-9]*),([0-9]*)')

# Set dtype for each group in regular expression.
# Any groups without a dtype defined for it will not be
# stored in numpy array
adapter.set_field_types(dict(zip(range(5), ['u4']*5)))

# Read all records
print(adapter[:])

