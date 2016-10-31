import pyaccumulo

conn = pyaccumulo.Accumulo('172.17.0.2', port=42424, user='root', password='secret')

def create_table(name, start, stop):
    global conn
    if conn.table_exists(name):
        conn.delete_table(name)
    conn.create_table(name)

    writer = conn.create_batch_writer(name)
    for i in range(start, stop):
        if name == 'uints':
            value = '{0:06d}'.format(i)
        elif name == 'ints':
            value = '{0:06d}'.format(i)
        elif name == 'floats':
            value = '{0:07f}'.format(i + 0.5)
        elif name == 'strings':
            value = 'xxx' + str(i)
        elif name == 'missing_data':
            if i % 2 == 0:
                value = 'NA'
            elif i % 3 == 0:
                value = 'nan'
            else:
                value = '{0:06d}'.format(i)
        else:
            raise ValueError('invalid table name')
        m = pyaccumulo.Mutation('row{0:06d}'.format(i - start))
        m.put(cf='f{0:06d}'.format(i - start), cq='q{0:06d}'.format(i - start), val=value)
        writer.add_mutation(m)
    writer.close()

create_table('uints', 0, 100000)
create_table('ints', -50000, 50000)
create_table('floats', -50000, 50000)
create_table('strings', 0, 100000)
create_table('missing_data', 0, 12)

conn.close()
