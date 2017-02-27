#ifndef FIELD_INFO_H
#define FIELD_INFO_H

#include "converter_functions.h"


typedef struct missing_values_t
{
    char **missing_values;
    uint32_t *missing_value_lens;
    uint32_t num_missing_values;
} MissingValues;


typedef struct fill_value_t
{
    void *fill_value;
    int loose;
} FillValue;


typedef struct field_info_t
{
    char *name;

    /* converter function to convert data to target data type */
    converter_func_ptr converter;
    void *converter_arg;

    MissingValues missing_values;

    FillValue fill_value;

    /* field width for fixed width data */
    uint32_t input_field_width;

    /* field size in output array */
    uint32_t output_field_size;

    /* flag allows user to fix the type. default, though, is to infer_type */
    int infer_type;

} FieldInfo;


typedef struct field_list_t
{
    uint32_t num_fields;
    FieldInfo *field_info;
} FieldList;


void clear_fields(FieldList *fields);
void set_num_fields(FieldList *fields, uint32_t num_fields);

void clear_missing_values(MissingValues *missing_values);
void clear_fill_value(FillValue *fill_value);

void init_missing_values(FieldList *fields, char *field_name,
    uint32_t field_num, uint32_t num_missing_values);

void add_missing_value(FieldList *fields, char *field_name,
    uint32_t field_num, char *missing_value, uint32_t missing_value_len);

void set_fill_value(FieldList *fields, char *field_name,
    uint32_t field_num, void *fill_value, uint32_t fill_value_len, int loose);

uint32_t get_field_size(FieldList *fields, char *field_name,
    uint32_t field_num);
uint32_t get_output_record_size(FieldList *fields);

void set_field_width(FieldList *fields, uint32_t field, uint32_t width);

/* Resets converter function pointers to null */
void reset_converters(FieldList *fields);

/* Sets converter function for specified field with specified field size.
 * converter_arg will be passed to converter function when called. */
void set_converter(FieldList *fields, uint32_t field_num, char *field_name,
    uint32_t output_field_size, converter_func_ptr converter,
    void *converter_arg);

/* Initialize the type of each of the fields to be inferred */
void init_infer_types(FieldList *fields);

int infer_types(FieldList *fields);

#endif
