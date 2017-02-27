import psycopg2
import numpy as np
import string

CASTS_TEST_NUM_RECORDS = 23456

if __name__ == '__main__':
    conn = psycopg2.connect('host=localhost dbname=postgres user=jayvius')
    conn.set_isolation_level(0)
    cursor = conn.cursor()
    cursor.execute('create database unit_tests')
    conn.set_isolation_level(1)
    conn.close()

    conn = psycopg2.connect('host=localhost dbname=unit_tests user=jayvius')

    cursor = conn.cursor()
    cursor.execute('CREATE EXTENSION postgis')
    cursor.execute('CREATE EXTENSION postgis_topology')

    cursor.execute('create table ints_test (int2 smallint, int4 integer, int8 bigint)')
    cmd = 'insert into ints_test (int2, int4, int8) values ({0}, {1}, {2})'
    cursor.execute(cmd.format(np.iinfo(np.int16).min, np.iinfo(np.int32).min, np.iinfo(np.int64).min))
    cursor.execute(cmd.format(0, 0, 0))
    cursor.execute(cmd.format(np.iinfo(np.int16).max, np.iinfo(np.int32).max, np.iinfo(np.int64).max))

    cursor.execute('create table serial_test (int2 smallserial, int4 serial, int8 bigserial)')
    cmd = 'insert into serial_test (int2, int4, int8) values ({0}, {1}, {2})'
    cursor.execute(cmd.format(np.iinfo(np.int16).min, np.iinfo(np.int32).min, np.iinfo(np.int64).min))
    cursor.execute(cmd.format(0, 0, 0))
    cursor.execute(cmd.format(np.iinfo(np.int16).max, np.iinfo(np.int32).max, np.iinfo(np.int64).max))

    cursor.execute('create table floats_test (float4 real, float8 double precision)')
    cmd = 'insert into floats_test (float4, float8) values ({0}, {1})'
    cursor.execute(cmd.format(np.finfo(np.float32).min, np.finfo(np.float64).min))
    cursor.execute(cmd.format(0.0, 0.0))
    cursor.execute(cmd.format(-1.1, 1.1))

    cursor.execute('create table numeric_test (numeric1 numeric(20, 10), numeric2 decimal(20, 10))')
    cmd = 'insert into numeric_test (numeric1, numeric2) values ({0}, {1})'
    cursor.execute(cmd.format(1234567890.0123456789, 1234567890.0123456789))

    cursor.execute('create table fixed_strings_test (fixed char(10))')
    cmd = "insert into fixed_strings_test (fixed) values ('{0}')"
    cursor.execute(cmd.format('aaa'))
    cursor.execute(cmd.format('bbb'))
    cursor.execute(cmd.format('ccc'))

    cursor.execute('create table var_strings_test (varchar varchar(10), text text)')
    cmd = "insert into var_strings_test (varchar, text) values ('{0}', '{1}')"
    cursor.execute(cmd.format('aaa', string.ascii_lowercase))
    cursor.execute(cmd.format('bbb', string.ascii_uppercase))
    cursor.execute(cmd.format('ccc', string.ascii_letters))

    cursor.execute('create table unicode_strings_test (fixed char(10), text text)')
    cursor.execute(u"insert into unicode_strings_test (fixed, text) values ('\u4242xxx', 'xxx\u4242')")

    cursor.execute(u'create table unicode_table_name_test (name\u4242 text)')
    cursor.execute(u"insert into unicode_table_name_test (name\u4242) values ('foo')")

    cursor.execute('create table geometric_test (point point, line line, '
                 'lseg lseg, box box, path path, polygon polygon, circle circle)')
    cursor.execute("insert into geometric_test (point, line, lseg, box, path, "
                 "polygon, circle) values "
                 "('(1.1 , 2.2)', '{1, 2, 3}', '((1, 2), (3, 4))', "
                 "'((1, 2), (3, 4))', '((1, 2), (3, 4), (5, 6))', "
                 "'((1, 2), (3, 4), (5, 6))', '((1, 2), 3)')")

    cursor.execute('create table casts_test (char char(10), int4 int4, float8 double precision)')
    cmd = "insert into casts_test (char, int4, float8) values ('{0}', {0}, {0}.{0})"
    for i in range(CASTS_TEST_NUM_RECORDS):
        cursor.execute(cmd.format(i))

    cursor.execute('create table missing_values_test (char char(5), int4 int4, float4 real, '
                 'point point, path path, polygon polygon)')
                 #'gis_point geometry(POINT), gis_multipoint geometry(MULTIPOINT), '
                 #'gis_polygon geometry(POLYGON))')
    cursor.execute('insert into missing_values_test default values')

    cursor.execute('create table empty_test (dummy int4)')

    cursor.execute('create table points ('
                 'point2d geometry(POINT), '
                 'point3d geometry(POINTZ), '
                 'point4d geometry(POINTZM))')
    cursor.execute("insert into points (point2d, point3d, point4d) values ("
                 "ST_GeomFromText('POINT(0 1)'), "
                 "ST_GeomFromText('POINT(0 1 2)'), "
                 "ST_GeomFromText('POINT(0 1 2 3)'))")

    cursor.execute('create table multipoints ('
                 'point2d geometry(MULTIPOINT), '
                 'point3d geometry(MULTIPOINTZ), '
                 'point4d geometry(MULTIPOINTZM))')
    cursor.execute("insert into multipoints (point2d, point3d, point4d) values ("
                 "ST_GeomFromText('MULTIPOINT(0 1, 2 3)'), "
                 "ST_GeomFromText('MULTIPOINT(0 1 2, 3 4 5)'), "
                 "ST_GeomFromText('MULTIPOINT(0 1 2 3, 4 5 6 7)'))")

    cursor.execute('create table lines ('
                 'line2d geometry(LINESTRING), '
                 'line3d geometry(LINESTRINGZ), '
                 'line4d geometry(LINESTRINGZM))')
    cursor.execute("insert into lines (line2d, line3d, line4d) values ("
                 "ST_GeomFromText('LINESTRING(0 1, 2 3)'), "
                 "ST_GeomFromText('LINESTRING(0 1 2, 3 4 5)'), "
                 "ST_GeomFromText('LINESTRING(0 1 2 3, 4 5 6 7)'))")
    cursor.execute("insert into lines (line2d, line3d, line4d) values ("
                 "ST_GeomFromText('LINESTRING(0 1, 2 3, 4 5)'), "
                 "ST_GeomFromText('LINESTRING(0 1 2, 3 4 5, 6 7 8)'), "
                 "ST_GeomFromText('LINESTRING(0 1 2 3, 4 5 6 7)'))")

    cursor.execute('create table multilines ('
                 'line2d geometry(MULTILINESTRING), '
                 'line3d geometry(MULTILINESTRINGZ), '
                 'line4d geometry(MULTILINESTRINGZM))')
    cursor.execute("insert into multilines (line2d, line3d, line4d) values ("
                 "ST_GeomFromText('MULTILINESTRING((0 1, 2 3), (4 5, 6 7))'), "
                 "ST_GeomFromText('MULTILINESTRING((0 1 2, 3 4 5), (6 7 8, 9 10 11, 12 13 14))'), "
                 "ST_GeomFromText('MULTILINESTRING((0 1 2 3, 4 5 6 7), (8 9 10 11, 12 13 14 15))'))")

    cursor.execute('create table polygons ('
                 'polygon2d geometry(POLYGON), '
                 'polygon3d geometry(POLYGONZ), '
                 'polygon4d geometry(POLYGONZM))')
    cursor.execute("insert into polygons (polygon2d, polygon3d, polygon4d) values ("
                 "ST_GeomFromText('POLYGON((0 1, 2 3, 4 5, 0 1), "
                                          "(0 1, 2 3, 4 5, 0 1), "
                                          "(0 1, 2 3, 4 5, 0 1))'), "
                 "ST_GeomFromText('POLYGON((0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                          "(0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                          "(0 1 2, 3 4 5, 6 7 8, 0 1 2))'), "
                 "ST_GeomFromText('POLYGON((0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                          "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                          "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3))'))")

    cursor.execute('create table multipolygons ('
                 'polygon2d geometry(MULTIPOLYGON), '
                 'polygon3d geometry(MULTIPOLYGONZ), '
                 'polygon4d geometry(MULTIPOLYGONZM))')
    cursor.execute("insert into multipolygons (polygon2d, polygon3d, polygon4d) values ("
                 "ST_GeomFromText('MULTIPOLYGON(((0 1, 2 3, 4 5, 0 1), "
                                                "(0 1, 2 3, 4 5, 0 1), "
                                                "(0 1, 2 3, 4 5, 0 1)), "
                                               "((0 1, 2 3, 4 5, 0 1), "
                                                "(0 1, 2 3, 4 5, 0 1), "
                                                "(0 1, 2 3, 4 5, 0 1)))'), "
                 "ST_GeomFromText('MULTIPOLYGON(((0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                                "(0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                                "(0 1 2, 3 4 5, 6 7 8, 0 1 2)), "
                                               "((0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                                "(0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                                "(0 1 2, 3 4 5, 6 7 8, 0 1 2)))'), "
                 "ST_GeomFromText('MULTIPOLYGON(((0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                                "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                                "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3)), "
                                               "((0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                                "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                                "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3)))'))")

    cursor.execute('create table triangles '
                 '(tri2d geometry(TRIANGLE), tri3d geometry(TRIANGLEZ), tri4d geometry(TRIANGLEZM))')
    cursor.execute("insert into triangles (tri2d, tri3d, tri4d) values ("
                 "ST_GeomFromText('TRIANGLE((0 0,1 1,2 2,0 0))'), "
                 "ST_GeomFromText('TRIANGLE((0 0 0,1 1 1,2 2 2,0 0 0))'), "
                 "ST_GeomFromText('TRIANGLE((0 0 0 0,1 1 1 1,2 2 2 2,0 0 0 0))'))")

    conn.commit()
    cursor.close()
    conn.close()
