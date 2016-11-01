from iopro import PostgresAdapter
import unittest
import testing.postgresql
import pandas as pd
import numpy as np
import sqlalchemy
import string
import pytest

class TestPostGIS(unittest.TestCase):
    
    def setUp(self):
        self.postgresql = testing.postgresql.Postgresql()
        engine = sqlalchemy.create_engine(self.postgresql.url())
        conn = engine.connect()

        conn.execute('CREATE EXTENSION postgis')
        conn.execute('CREATE EXTENSION postgis_topology')
        #conn.execute('CREATE EXTENSION fuzzystrmatch')
        #conn.execute('CREATE EXTENSION postgis_tiger_geocoder')

        conn.execute('create table points ('
                     'point2d geometry(POINT), '
                     'point3d geometry(POINTZ), '
                     'point4d geometry(POINTZM))')
        conn.execute("insert into points (point2d, point3d, point4d) values ("
                     "ST_GeomFromText('POINT(0 1)'), "
                     "ST_GeomFromText('POINT(0 1 2)'), "
                     "ST_GeomFromText('POINT(0 1 2 3)'))")

        conn.execute('create table multipoints ('
                     'point2d geometry(MULTIPOINT), '
                     'point3d geometry(MULTIPOINTZ), '
                     'point4d geometry(MULTIPOINTZM))')
        conn.execute("insert into multipoints (point2d, point3d, point4d) values ("
                     "ST_GeomFromText('MULTIPOINT(0 1, 2 3)'), "
                     "ST_GeomFromText('MULTIPOINT(0 1 2, 3 4 5)'), "
                     "ST_GeomFromText('MULTIPOINT(0 1 2 3, 4 5 6 7)'))")

        conn.execute('create table lines ('
                     'line2d geometry(LINESTRING), '
                     'line3d geometry(LINESTRINGZ), '
                     'line4d geometry(LINESTRINGZM))')
        conn.execute("insert into lines (line2d, line3d, line4d) values ("
                     "ST_GeomFromText('LINESTRING(0 1, 2 3)'), "
                     "ST_GeomFromText('LINESTRING(0 1 2, 3 4 5)'), "
                     "ST_GeomFromText('LINESTRING(0 1 2 3, 4 5 6 7)'))")
        conn.execute("insert into lines (line2d, line3d, line4d) values ("
                     "ST_GeomFromText('LINESTRING(0 1, 2 3, 4 5)'), "
                     "ST_GeomFromText('LINESTRING(0 1 2, 3 4 5, 6 7 8)'), "
                     "ST_GeomFromText('LINESTRING(0 1 2 3, 4 5 6 7)'))")

        conn.execute('create table multilines ('
                     'line2d geometry(MULTILINESTRING), '
                     'line3d geometry(MULTILINESTRINGZ), '
                     'line4d geometry(MULTILINESTRINGZM))')
        conn.execute("insert into multilines (line2d, line3d, line4d) values ("
                     "ST_GeomFromText('MULTILINESTRING((0 1, 2 3), (4 5, 6 7))'), "
                     "ST_GeomFromText('MULTILINESTRING((0 1 2, 3 4 5), (6 7 8, 9 10 11, 12 13 14))'), "
                     "ST_GeomFromText('MULTILINESTRING((0 1 2 3, 4 5 6 7), (8 9 10 11, 12 13 14 15))'))")

        conn.execute('create table polygons ('
                     'polygon2d geometry(POLYGON), '
                     'polygon3d geometry(POLYGONZ), '
                     'polygon4d geometry(POLYGONZM))')
        conn.execute("insert into polygons (polygon2d, polygon3d, polygon4d) values ("
                     "ST_GeomFromText('POLYGON((0 1, 2 3, 4 5, 0 1), "
                                              "(0 1, 2 3, 4 5, 0 1), "
                                              "(0 1, 2 3, 4 5, 0 1))'), "
                     "ST_GeomFromText('POLYGON((0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                              "(0 1 2, 3 4 5, 6 7 8, 0 1 2), "
                                              "(0 1 2, 3 4 5, 6 7 8, 0 1 2))'), "
                     "ST_GeomFromText('POLYGON((0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                              "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3), "
                                              "(0 1 2 3, 4 5 6 7, 8 9 10 11, 0 1 2 3))'))")

        conn.execute('create table multipolygons ('
                     'polygon2d geometry(MULTIPOLYGON), '
                     'polygon3d geometry(MULTIPOLYGONZ), '
                     'polygon4d geometry(MULTIPOLYGONZM))')
        conn.execute("insert into multipolygons (polygon2d, polygon3d, polygon4d) values ("
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

        conn.execute('create table triangles '
                     '(tri2d geometry(TRIANGLE), tri3d geometry(TRIANGLEZ), tri4d geometry(TRIANGLEZM))')
        conn.execute("insert into triangles (tri2d, tri3d, tri4d) values ("
                     "ST_GeomFromText('TRIANGLE((0 0,1 1,2 2,0 0))'), "
                     "ST_GeomFromText('TRIANGLE((0 0 0,1 1 1,2 2 2,0 0 0))'), "
                     "ST_GeomFromText('TRIANGLE((0 0 0 0,1 1 1 1,2 2 2 2,0 0 0 0))'))")

    def test_points(self):
        adapter = PostgresAdapter(self.postgresql.url(),
            query='select point2d, point3d, point4d from points')
        result = adapter[:]
        expected = np.array([(u'POINT (0.000000 1.000000)',
                              u'POINT (0.000000 1.000000 2.000000)',
                              u'POINT (0.000000 1.000000 2.000000 3.000000)')],
                            dtype=[('point2d', 'O'),
                                   ('point3d', 'O'),
                                   ('point4d', 'O')])
        np.testing.assert_array_equal(expected, result)

        adapter.field_types = ['f8', 'O', 'f8']
        result = adapter[:]
        expected = np.array([([0.0, 1.0],
                              u'POINT (0.000000 1.000000 2.000000)',
                              [0.0, 1.0, 2.0, 3.0])],
                            dtype=[('point2d', 'f8', (2,)),
                                   ('point3d', 'O'),
                                   ('point4d', 'f8', (4,))])
        np.testing.assert_array_equal(expected, result)

    def test_multipoints(self):
        adapter = PostgresAdapter(self.postgresql.url(),
            query='select point2d, point3d, point4d from multipoints')
        adapter.field_shapes = {'point2d': 1, 'point3d': 4}
        result = adapter[:]
        expected = np.array([([[0, 1]],
                              [[0, 1, 2], [3, 4, 5], [0, 0, 0], [0, 0, 0]],
                              'MULTIPOINT ((0.000000 1.000000 2.000000 3.000000), '
                                          '(4.000000 5.000000 6.000000 7.000000))')],
                            dtype=[('point2d', 'f8', (1, 2)),
                                   ('point3d', 'f8', (4, 3)),
                                   ('point4d', 'O')])
        np.testing.assert_array_equal(expected, result)

    def test_lines(self):
        adapter = PostgresAdapter(self.postgresql.url(),
            query='select line2d, line3d, line4d from lines')
        adapter.field_shapes = {'line2d': 1, 'line3d': 3}
        result = adapter[:]
        expected = np.array([([[0.0, 1.0]],
                              [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]],
                              'LINESTRING (0.000000 1.000000 2.000000 3.000000, '
                                          '4.000000 5.000000 6.000000 7.000000)'),
                             ([[0.0, 1.0]],
                              [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                              'LINESTRING (0.000000 1.000000 2.000000 3.000000, '
                                          '4.000000 5.000000 6.000000 7.000000)')],
                            dtype=[('line2d', 'f8', (1,2)),
                                   ('line3d', 'f8', (3,3)),
                                   ('line4d', 'O')])
        np.testing.assert_array_equal(expected, result)
        
    def test_multilines(self):
        adapter = PostgresAdapter(self.postgresql.url(),
            query='select line2d, line3d, line4d from multilines')
        adapter.field_shapes = {'line3d': (2, 3), 'line4d': (2, 2)}
        result = adapter[:]
        expected = np.array([('MULTILINESTRING ((0.000000 1.000000, 2.000000 3.000000), '
                                               '(4.000000 5.000000, 6.000000 7.000000))',
                              [[[0, 1, 2], [3, 4, 5], [0, 0, 0]], [[6, 7, 8], [9, 10, 11], [12, 13, 14]]],
                              [[(0, 1, 2, 3), (4, 5, 6, 7)], [(8, 9, 10, 11), (12, 13, 14, 15)]])],
                            dtype=[('line2d', 'O'),
                                   ('line3d', 'f8', (2, 3, 3)),
                                   ('line4d', 'f8', (2, 2, 4))])
        np.testing.assert_array_equal(expected, result)
        
    def test_polygons(self):
        adapter = PostgresAdapter(self.postgresql.url(),
            query='select polygon2d, polygon3d, polygon4d from polygons')
        adapter.field_shapes = {'polygon3d': (4, 5), 'polygon4d': (3, 4)}
        result = adapter[:]
        expected = np.array([('POLYGON ((0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                       '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                       '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000))',
                              [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                              [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]],
                               [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]],
                               [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]]])],
                            dtype=[('polygon2d', 'O'),
                                   ('polygon3d', 'f8', (4, 5, 3)),
                                   ('polygon4d', 'f8', (3, 4, 4))])
        np.testing.assert_array_equal(expected, result)

    def test_multipolygons(self):
        adapter = PostgresAdapter(self.postgresql.url(),
            query='select polygon2d, polygon3d, polygon4d from multipolygons')
        adapter.field_shapes = {'polygon3d': (2, 4, 5), 'polygon4d': (2, 3, 4)}
        result = adapter[:]
        expected = np.array([('MULTIPOLYGON (((0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                             '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                             '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000)), '
                                            '((0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                             '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000), '
                                             '(0.000000 1.000000, 2.000000 3.000000, 4.000000 5.000000, 0.000000 1.000000)))',
                              [[[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                              [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 2], [0, 0, 0]],
                               [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
                              [[[(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                               [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                               [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)]],
                              [[(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                               [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)],
                               [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (0, 1, 2, 3)]]])],
                            dtype=[('polygon2d', 'O'),
                                   ('polygon3d', 'f8', (2, 4, 5, 3)),
                                   ('polygon4d', 'f8', (2, 3, 4, 4))])
        np.testing.assert_array_equal(expected, result)

    def test_dataframe(self):
        adapter = PostgresAdapter(self.postgresql.url(), dataframe=True, table='points')
        result = adapter[:]
        expected = pd.DataFrame(np.array([('POINT (0.000000 1.000000)',
                                           'POINT (0.000000 1.000000 2.000000)',
                                           'POINT (0.000000 1.000000 2.000000 3.000000)')],
                                dtype=[('point2d', 'O'),
                                       ('point3d', 'O'),
                                       ('point4d', 'O')]))
        np.testing.assert_array_equal(expected, result)

        adapter = PostgresAdapter(self.postgresql.url(), dataframe=True, table='multipoints')
        result = adapter[:]
        expected = pd.DataFrame(np.array([('MULTIPOINT ((0.000000 1.000000), (2.000000 3.000000))',
                                           'MULTIPOINT ((0.000000 1.000000 2.000000), (3.000000 4.000000 5.000000))',
                                           'MULTIPOINT ((0.000000 1.000000 2.000000 3.000000), '
                                                       '(4.000000 5.000000 6.000000 7.000000))')],
                                dtype=[('point2d', 'O'),
                                       ('point3d', 'O'),
                                       ('point4d', 'O')]))
        np.testing.assert_array_equal(expected, result)

    def tearDown(self):
        self.postgresql.stop()
    
def run(verbosity=2):
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPostGIS)
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)

if __name__ == '__main__':
    run()
