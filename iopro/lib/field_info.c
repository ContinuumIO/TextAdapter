#include "field_info.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>


/* Set the number of fields in input data. This  */
void set_num_fields(FieldList *fields, uint32_t num_fields)
{
    uint32_t i;

    #ifdef DEBUG_ADAPTER
    printf("set_num_fields() setting number of fields to %u\n", num_fields);
    #endif

    if (fields == NULL)
        return;

    if (fields->field_info != NULL)
    {
        clear_fields(fields);
    }

    if (num_fields > 0)
        fields->field_info = (FieldInfo*)calloc(num_fields, sizeof(FieldInfo));

    fields->num_fields = num_fields;

    for (i = 0; i < num_fields; i++)
    {
        fields->field_info[i].infer_type = 1;
    }
}

/* Initialize infer_type flag in each field to 1 */
void init_infer_types(FieldList *fields)
{
    uint32_t i;
    for(i = 0; i < fields->num_fields; i++)
    {
        fields->field_info[i].infer_type = 1;
    }
}

/* Initialize missing value struct */
void init_missing_values(FieldList *fields, char *field_name,
    uint32_t field_num, uint32_t num_missing_values)
{
    MissingValues *missing_values;

    if (fields == NULL)
        return;

    if (field_num >= fields->num_fields)
        return;

    missing_values = &fields->field_info[field_num].missing_values;

    clear_missing_values(missing_values);

    missing_values->num_missing_values = num_missing_values;
    missing_values->missing_value_lens =
        calloc(num_missing_values, sizeof(uint32_t));
    missing_values->missing_values =
        calloc(num_missing_values, sizeof(char *));
}


/* Add missing value string for the specified field */
void add_missing_value(FieldList *fields, char *field_name,
    uint32_t field_num, char *missing_value, uint32_t missing_value_len)
{
    MissingValues *missing_values;
    uint32_t i;

    if (fields == NULL)
        return;

    if (field_num >= fields->num_fields)
        return;

    missing_values = &fields->field_info[field_num].missing_values;

    /* Find first empty entry in missing values array to store missing
       value string */
    i = 0;
    while (i < missing_values->num_missing_values &&
           missing_values->missing_values[i] > 0)
    {
        i++;
    }

    missing_values->missing_values[i] =
        calloc(missing_value_len + 1, sizeof(char));
    strncpy(missing_values->missing_values[i], missing_value, missing_value_len);
    missing_values->missing_value_lens[i] = missing_value_len;
}


/* Set pointer to fill value for specified field. Positive valeu for
   'loose' argument enables fill value to be used when token for this
   field cannot be converted. */
void set_fill_value(FieldList *fields, char *field_name,
    uint32_t field_num, void *new_fill_value, uint32_t fill_value_len, int loose)
{
    FillValue *fill_value;

    if (fields == NULL)
        return;

    if (field_num >= fields->num_fields)
        return;

    fill_value = &fields->field_info[field_num].fill_value;

    if (new_fill_value == NULL)
    {
        clear_fill_value(fill_value);
    }
    else
    {
        fill_value->fill_value = calloc(1, fill_value_len);
        memcpy(fill_value->fill_value, new_fill_value, fill_value_len);
        fill_value->loose = loose;
    }
}


uint32_t get_field_size(FieldList *fields, char *field_name, uint32_t field_num)
{
    uint32_t i;

    if (fields == NULL)
        return 0;

    if (field_name != NULL)
    {
        i = 0;
        while (i < fields->num_fields)
        {
            if (strcpy(fields->field_info[i].name, field_name))
            {
                return fields->field_info[i].output_field_size;
            }
            i++;
        }

        return 0;
    }
    else
    {
        return fields->field_info[field_num].output_field_size;
    }
}


uint32_t get_output_record_size(FieldList *fields)
{
    uint32_t i;
    uint32_t rec_size;

    if (fields == NULL)
        return 0;

    rec_size = 0;

    for (i = 0; i < fields->num_fields; i++)
    {
        if (fields->field_info[i].converter != NULL)
        {
            rec_size += fields->field_info[i].output_field_size;
        }
    }

    return rec_size;
}



/* Deallocate missing value strings */
void clear_missing_values(MissingValues *missing_values)
{
    uint32_t i;

    assert(missing_values != NULL);

    if (missing_values->missing_values != NULL)
    {
        for (i = 0; i < missing_values->num_missing_values; i++)
        {
            if (missing_values->missing_values[i] != NULL)
                free(missing_values->missing_values[i]);
        }

        free(missing_values->missing_values);
        missing_values->missing_values = NULL;
    }

    if (missing_values->missing_value_lens != NULL)
    {
        free(missing_values->missing_value_lens);
        missing_values->missing_value_lens = NULL;
    }

    missing_values->num_missing_values = 0;
}


/* Deallocate pointer to fill value for specified field */
void clear_fill_value(FillValue *fill_value)
{
    assert(fill_value != NULL);

    if (fill_value->fill_value != NULL)
    {
        free(fill_value->fill_value);
        fill_value->fill_value = NULL;
    }
}


void clear_fields(FieldList *fields)
{
    uint32_t i;

    for (i = 0; i < fields->num_fields; i++)
    {
        if (fields->field_info[i].name != NULL)
        {
            free(fields->field_info[i].name);
        }
        fields->field_info[i].name = NULL;

        fields->field_info[i].converter = NULL;
        fields->field_info[i].converter_arg = NULL;

        clear_missing_values(&fields->field_info[i].missing_values);
        clear_fill_value(&fields->field_info[i].fill_value);

        fields->field_info[i].output_field_size = 0;
        fields->field_info[i].input_field_width = 0;
    }
    
    free(fields->field_info);
}


/* Set fixed field width for specified field */
void set_field_width(FieldList *fields, uint32_t field, uint32_t width)
{
    if (fields == NULL)
        return;

    if (field >= fields->num_fields)
        return;
     
    fields->field_info[field].input_field_width = width;
}


void reset_converters(FieldList *fields)
{
    uint32_t field;

    if (fields == NULL)
        return;

    for (field = 0; field < fields->num_fields; field++)
    {
        fields->field_info[field].converter = NULL;
        fields->field_info[field].converter_arg = NULL;
    }
}


void set_converter(FieldList *fields, uint32_t field_num, char *field_name,
    uint32_t output_field_size, converter_func_ptr converter,
    void *converter_arg)
{
    if (fields == NULL)
        return;

    if (field_num >= fields->num_fields)
        return;

    //if (field_name == NULL)
    //    return;

    if (fields->field_info[field_num].name != NULL)
    {
        free(fields->field_info[field_num].name);
    }

    if (field_name != NULL)
    {
        fields->field_info[field_num].name =
            calloc(strlen(field_name), sizeof(char));
        strncpy(fields->field_info[field_num].name, field_name, strlen(field_name));
    }
    else
    {
        fields->field_info[field_num].name = NULL;
    }

    fields->field_info[field_num].converter = converter;
    fields->field_info[field_num].converter_arg = converter_arg;
    fields->field_info[field_num].output_field_size = output_field_size;
}


int infer_types(FieldList *fields)
{
    uint32_t i;

    for (i = 0; i < fields->num_fields; i++)
    {
        if (fields->field_info[i].infer_type == 1)
            return 1;
    }

    return 0;
}
