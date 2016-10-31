import pandas as pd
import numpy as np

# copy foo1 (value1, value2) from '/Users/jayvius/Projects/iopro/iopro/postgresadapter/data.csv' delimiter ',' csv header;

num_records = 10
values1 = np.random.randint(0, 1000000, num_records)
values2 = np.random.rand(num_records)
data = pd.DataFrame({'values1': values1, 'values2':values2})
data.to_csv('data.csv', index=False)
