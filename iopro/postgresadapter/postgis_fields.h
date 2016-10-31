#ifndef POSTGIS_FIELDS_H
#define POSTGIS_FIELDS_H

#include <libpq-fe.h>
#include <_stdint.h>

/*
 * Values were figured out from observing data returned from PostGIS queries,
 * and Well Known Text (WKT) and Well Known Binary (WKB) descriptions
 * (https://en.wikipedia.org/wiki/Well-known_text)
 * (https://trac.osgeo.org/postgis/browser/trunk/doc/ZMSgeoms.txt)
 */
#define GIS_POINT2D 0x1
#define GIS_POINT3D 0x80000001
#define GIS_POINT4D 0xc0000001
#define GIS_LINE2D 0x2
#define GIS_LINE3D 0x80000002
#define GIS_LINE4D 0xc0000002
#define GIS_POLYGON2D 0x3
#define GIS_POLYGON3D 0x80000003
#define GIS_POLYGON4D 0xc0000003
#define GIS_MULTIPOINT2D 0x4
#define GIS_MULTIPOINT3D 0x80000004
#define GIS_MULTIPOINT4D 0xc0000004
#define GIS_MULTILINE2D 0x5
#define GIS_MULTILINE3D 0x80000005
#define GIS_MULTILINE4D 0xc0000005
#define GIS_MULTIPOLYGON2D 0x6
#define GIS_MULTIPOLYGON3D 0x80000006
#define GIS_MULTIPOLYGON4D 0xc0000006

uint32_t get_gis_type(char *data, int *has_srid);
int get_gis_point_size(uint32_t gis_type);
int parse_gis_data(char *data, int *field_shapes, char **outputi, int dataframe);

#endif
