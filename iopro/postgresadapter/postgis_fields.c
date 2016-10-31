#include "Python.h"
#include "postgis_fields.h"
#include <string.h>
#include <assert.h>
#include <kstring.h>

// GIS header consists of 1 byte endian flag, and 4 byte integer
// indicating number of geometry items
#define GIS_HEADER 5

void *create_list(void **);
void add_to_list(void *, double *, int);
void append_list(void *, void *);
void *convert_list_to_array(void *);
void create_string(void **output, const char *value);

uint32_t get_gis_type(char *data, int *has_srid)
{
    uint32_t gis_type;
    uint32_t SRID_FLAG = 0x20000000;

    assert(data);

    memcpy(&gis_type, data + 1, sizeof(gis_type));

    /* Extract SRID presence flag */
    if (has_srid) {
      *has_srid = SRID_FLAG & gis_type;
    }
    /* And out the SRID-presence flag for comparing type: see https://trac.osgeo.org/postgis/browser/trunk/doc/ZMSgeoms.txt */
    gis_type = ~SRID_FLAG & gis_type;
    return gis_type;
}

int get_gis_point_size(uint32_t gis_type)
{
    switch (gis_type) {
        case GIS_POINT2D:
        case GIS_LINE2D:
        case GIS_POLYGON2D:
        case GIS_MULTIPOINT2D:
        case GIS_MULTILINE2D:
        case GIS_MULTIPOLYGON2D:
            return 2;
        case GIS_POINT3D:
        case GIS_LINE3D:
        case GIS_POLYGON3D:
        case GIS_MULTIPOINT3D:
        case GIS_MULTILINE3D:
        case GIS_MULTIPOLYGON3D:
            return 3;
        case GIS_POINT4D:
        case GIS_LINE4D:
        case GIS_POLYGON4D:
        case GIS_MULTIPOINT4D:
        case GIS_MULTILINE4D:
        case GIS_MULTIPOLYGON4D:
            return 4;
        default:
            return -1;
    };
}

int parse_points_as_text(char *data, int num_points, int point_size, kstring_t *ks)
{
    int i, j;

    assert(data);
    assert(ks);

    for (i = 0; i < num_points; i++) {
        for (j = 0; j < point_size; j++) {
            ksprintf(ks, "%f", ((double*)data)[j]);
            if (j + 1 < point_size) {
                kputs(" ", ks);
            }
        }
        if (i + 1 < num_points) {
            kputs(", ", ks);
        }
        data += point_size * sizeof(double);
    }
    
    return sizeof(double) * point_size * num_points;
}

int parse_points_as_floats(char *data, int num_points, int point_size, char **output)
{
    assert(data);
    assert(output);
    assert(*output);
    
    memcpy(*output, data, sizeof(double) * point_size * num_points);
    *output += num_points * point_size * sizeof(double);

    return sizeof(double) * point_size * num_points;
}

int parse_line_as_text(char *data, int point_size, kstring_t *ks)
{
    uint32_t num_points;
    int result;

    assert(data);
    assert(ks);
    
    memcpy(&num_points, data + GIS_HEADER, sizeof(num_points));
    
    kputs("(", ks);
    result = parse_points_as_text(data + GIS_HEADER + sizeof(num_points),
                                  num_points,
                                  point_size,
                                  ks);
    if (!result) {
        return 0;
    }
    kputs(")", ks);

    return GIS_HEADER + sizeof(num_points) + num_points * point_size * sizeof(double);
}

int parse_line_as_floats(char *data, int point_size, int *field_shape, char **output)
{
    uint32_t num_points;
    int result;

    memcpy(&num_points, data + GIS_HEADER, sizeof(num_points));

    assert(data);
    assert(field_shape);
    assert(output);
    assert(*output);
    
    if (num_points > field_shape[0]) {
        num_points = field_shape[0];
    }
    result = parse_points_as_floats(data + GIS_HEADER + sizeof(num_points),
                                    num_points,
                                    point_size,
                                    output);
    if (!result) {
        return 0;
    }
    *output += (field_shape[0] - num_points) * point_size * sizeof(double);

    return GIS_HEADER + sizeof(num_points) + num_points * point_size * sizeof(double);
}

int parse_polygon_as_text(char *data, int point_size, kstring_t *ks)
{
    uint32_t num_rings;
    int offset;
    int result;
    int i;
    uint32_t num_points;

    memcpy(&num_rings, data + GIS_HEADER, sizeof(num_rings));
    offset = GIS_HEADER + sizeof(num_rings);

    kputs("(", ks);
    for (i = 0; i < num_rings; i++) {
        memcpy(&num_points, data + offset, sizeof(num_points));
        kputs("(", ks);
        result = parse_points_as_text(data + offset + sizeof(num_points), num_points, point_size, ks);
        if (!result) {
            return 0;
        }
        kputs(")", ks);
        if (i + 1 < num_rings) {
            kputs(", ", ks);
        }
        offset += sizeof(num_points) + num_points * point_size * sizeof(double);
    }
    kputs(")", ks);

    return offset;
}

int parse_polygon_as_floats(char *data, int point_size, int *field_shape, char **output)
{
    uint32_t num_rings;
    int offset = 9;
    int result;
    int i;
    uint32_t num_points;

    assert(data);
    assert(field_shape);
    assert(output);
    assert(*output);

    memcpy(&num_rings, data + GIS_HEADER, sizeof(num_rings));
    offset = GIS_HEADER + sizeof(num_rings);
    
    if (num_rings > field_shape[0]) {
        num_rings = field_shape[0];
    }

    for (i = 0; i < num_rings; i++) {
        memcpy(&num_points, data + offset, sizeof(num_points));
        if (num_points > field_shape[1]) {
            num_points = field_shape[1];
        }
        result = parse_points_as_floats(data + offset + sizeof(num_points), num_points, point_size, output);
        if (!result) {
            return 0;
        }
        *output += (field_shape[1] - num_points) * point_size * sizeof(double);
        offset += sizeof(num_points) + num_points * point_size * sizeof(double);
    }
    
    *output += (field_shape[0] - num_rings) * field_shape[1] * point_size * sizeof(double);
    return offset;
}

int parse_multipoint_as_text(char *data, int point_size, kstring_t *ks)
{
    uint32_t num_points;
    int result;
    int offset;
    int i;

    assert(data);
    assert(ks);

    memcpy(&num_points, data + GIS_HEADER, sizeof(num_points));
    offset = GIS_HEADER + sizeof(num_points);

    kputs("(", ks);
    for (i = 0; i < num_points; i++) {
        kputs("(", ks);
        offset += GIS_HEADER;
        result = parse_points_as_text(data + offset, 1, point_size, ks);
        if (!result) {
            return 0;
        }
        kputs(")", ks);
        offset += point_size * sizeof(double);
        if (i + 1 < num_points) {
            kputs(", ", ks);
        }
    }
    kputs(")", ks);

    return offset;
}

int parse_multipoint_as_floats(char *data, int point_size, int *field_shape, char **output)
{
    uint32_t num_points;
    int result;
    int offset;
    int i;

    assert(data);
    assert(field_shape);
    assert(output);
    assert(*output);
    
    memcpy(&num_points, data + GIS_HEADER, sizeof(num_points));
    offset = GIS_HEADER + sizeof(num_points);

    if (num_points > field_shape[0]) {
        num_points = field_shape[0];
    }

    for (i = 0; i < num_points; i++) {
        offset += GIS_HEADER;
        result = parse_points_as_floats(data + offset, 1, point_size, output);
        if (!result) {
            return 0;
        }
        offset += point_size * sizeof(double);
    }
    *output += (field_shape[0] - num_points) * point_size * sizeof(double);

    return offset;
}

int parse_multiline_as_text(char *data, int point_size, kstring_t *ks)
{
    uint32_t num_lines;
    int result;
    int offset;
    int i;

    assert(data);
    assert(ks);

    memcpy(&num_lines, data + GIS_HEADER, sizeof(num_lines));
    offset = GIS_HEADER + sizeof(num_lines);
    
    kputs("(", ks);
    for (i = 0; i < num_lines; i++) {
        result = parse_line_as_text(data + offset, point_size, ks);
        if (!result) {
            return 0;
        }
        if (i + 1 < num_lines) {
            kputs(", ", ks);
        }
        offset += result;
    }
    kputs(")", ks);

    return offset;
}

int parse_multiline_as_floats(char *data, int point_size, int *field_shape, char **output)
{
    uint32_t num_lines;
    int result;
    int offset;
    int i;

    assert(data);
    assert(field_shape);
    assert(output);
    assert(*output);

    memcpy(&num_lines, data + GIS_HEADER, sizeof(num_lines));
    offset = GIS_HEADER + sizeof(num_lines);

    if (num_lines > field_shape[0]) {
        num_lines = field_shape[0];
    }

    for (i = 0; i < num_lines; i++) {
        result = parse_line_as_floats(data + offset, point_size, field_shape + 1, output);
        if (!result) {
            return 0;
        }
        offset += result;
    }

    return offset;
}

int parse_multipolygon_as_text(char *data, int point_size, kstring_t *ks)
{
    uint32_t num_polygons;
    int result;
    int offset;
    int i;

    assert(data);
    assert(ks);

    memcpy(&num_polygons, data + GIS_HEADER, sizeof(num_polygons));
    offset = GIS_HEADER + sizeof(num_polygons);

    kputs("(", ks);
    for (i = 0; i < num_polygons; i++) {
        result = parse_polygon_as_text(data + offset, point_size, ks);
        if (!result) {
            return 0;
        }
        offset += result;
        if (i + 1 < num_polygons) {
            kputs(", ", ks);
        }
    }
    kputs(")", ks);

    return offset;
}

int parse_multipolygon_as_floats(char *data, int point_size, int *field_shape, char **output)
{
    uint32_t num_polygons;
    int result;
    int offset;
    int i;

    assert(dat);
    assert(field_shape);
    assert(output);
    assert(*output);

    memcpy(&num_polygons, data + GIS_HEADER, sizeof(num_polygons));
    offset = GIS_HEADER + sizeof(num_polygons);

    if (num_polygons > field_shape[0]) {
        num_polygons = field_shape[0];
    }

    for (i = 0; i < num_polygons; i++) {
        result = parse_polygon_as_floats(data + offset, point_size, field_shape + 1, output);
        if (!result) {
            return 0;
        }
        offset += result;
    }

    *output += (field_shape[0] - num_polygons) * field_shape[1] * field_shape[2] * point_size * sizeof(double);

    return offset;
}

int parse_gis_data(char *data, int *field_shape, char **output, int dataframe)
{
    int has_srid;
    uint32_t type = get_gis_type(data, &has_srid);
    int point_size = get_gis_point_size(type);
    int result = 0;
    kstring_t ks;

    /* for now, don't extract SRID, but gracefully ignore the 4-byte SRID integer after the type */
    if (has_srid) {
      data += 4;
    }

    ks.l = 0;
    ks.m = 0;
    ks.s = NULL;

    switch (type)
    {
        case GIS_POINT2D:
        case GIS_POINT3D:
        case GIS_POINT4D:
            if (field_shape == NULL) {
                kputs("POINT (", &ks);
                result = parse_points_as_text(data + GIS_HEADER, 1, point_size, &ks);
                kputs(")", &ks);
            }
            else {
                result = parse_points_as_floats(data + GIS_HEADER, 1, point_size, output);
            }
            break;
        case GIS_LINE2D:
        case GIS_LINE3D:
        case GIS_LINE4D:
            if (field_shape == NULL) {
                kputs("LINESTRING ", &ks);
                result = parse_line_as_text(data, point_size, &ks);
            }
            else {
                result = parse_line_as_floats(data, point_size, field_shape, output);
            }
            break;
        case GIS_POLYGON2D:
        case GIS_POLYGON3D:
        case GIS_POLYGON4D:
            if (field_shape == NULL) {
                kputs("POLYGON ", &ks);
                result = parse_polygon_as_text(data, point_size, &ks);
            }
            else {
                result = parse_polygon_as_floats(data, point_size, field_shape, output);
            }
            break;
        case GIS_MULTIPOINT2D:
        case GIS_MULTIPOINT3D:
        case GIS_MULTIPOINT4D:
            if (field_shape == NULL) {
                kputs("MULTIPOINT ", &ks);
                result = parse_multipoint_as_text(data, point_size, &ks);
            }
            else {
                result = parse_multipoint_as_floats(data, point_size, field_shape, output);
            }
            break;
        case GIS_MULTILINE2D:
        case GIS_MULTILINE3D:
        case GIS_MULTILINE4D:
            if (field_shape == NULL) {
                kputs("MULTILINESTRING ", &ks);
                result = parse_multiline_as_text(data, point_size, &ks);
            }
            else {
                result = parse_multiline_as_floats(data, point_size, field_shape, output);
            }
            break;
        case GIS_MULTIPOLYGON2D:
        case GIS_MULTIPOLYGON3D:
        case GIS_MULTIPOLYGON4D:
            if (field_shape == NULL) {
                kputs("MULTIPOLYGON ", &ks);
                result = parse_multipolygon_as_text(data, point_size, &ks);
            }
            else {
                result = parse_multipolygon_as_floats(data, point_size, field_shape, output);
            }
            break;
    };

    if (field_shape == NULL) {
        create_string(output, ks.s);
        free(ks.s);
    }

    return result;
}
