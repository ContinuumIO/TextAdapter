#!/usr/bin/python

import time
import gzip
import numpy


def generate_dataset(output, valueIter, delimiter, num_recs):
    for i in range(0, num_recs):
        line = ''
        for j in range(0, 5):
            if j == 5 - 1:
                line += str(valueIter.next())
            else:
                line += str(valueIter.next()) + delimiter
        output.write(line)
        output.write('\n')
    output.seek(0)


class IntIter(object):

    def __init__(self):
        self.value = 0

    def __str__(self):
        return 'ints'

    def __iter__(self):
        return self

    def next(self):
        nextValue = self.value
        self.value = self.value + 1
        return nextValue


class SignedIntIter(object):

    def __init__(self):
        self.value = -1

    def __str__(self):
        return 'signed int'

    def __iter__(self):
        return self

    def next(self):
        nextValue = self.value
        if self.value < 0:
            self.value = self.value - 1
        else:
            self.value = self.value + 1
        self.value *= -1
        return nextValue


class FloatIter(object):

    def __init__(self):
        self.value = 0.0

    def __str__(self):
        return 'floats'

    def __iter__(self):
        return self

    def next(self):
        nextValue = self.value
        self.value = self.value + 0.1
        return nextValue


class MissingValuesIter(object):

    def __init__(self):
        self.value = 0

    def __str__(self):
        return 'missing values'

    def __iter__(self):
        return self

    def next(self):
        nextValue = self.value
        if nextValue % 20 == 0:
            nextValue = 'NA'
        elif nextValue % 20 == 4:
            nextValue = 'xx'
        elif nextValue % 20 == 5:
            nextValue = 'NaN'
        elif nextValue % 20 == 9:
            nextValue = 'inf'
        self.value = self.value + 1
        return nextValue


class FixedWidthIter(object):
    
    def __init__(self):
        self.field = 0
        self.fieldValues = ['00','000','0000','00000','000000']

    def __str__(self):
        return 'fixed widths'

    def __iter__(self):
        return self

    def next(self):
        nextValue = self.fieldValues[self.field]

        self.field = self.field + 1
        if self.field == 5:
            self.field = 0
            self.fieldValues[0] = str((int(self.fieldValues[0]) + 1) % 100).zfill(2)
            self.fieldValues[1] = str((int(self.fieldValues[1]) + 1) % 1000).zfill(3)
            self.fieldValues[2] = str((int(self.fieldValues[2]) + 1) % 10000).zfill(4)
            self.fieldValues[3] = str((int(self.fieldValues[3]) + 1) % 100000).zfill(5)
            self.fieldValues[4] = str((int(self.fieldValues[4]) + 1) % 1000000).zfill(6)

        return nextValue


class QuoteIter(object):

    def __init__(self):
        self.value = 0

    def __str__(self):
        return 'quoted strings'

    def __iter__(self):
        return self

    def next(self):
        nextValue = self.value
        characters = list(str(nextValue))
        nextValue = '"' + ',\n'.join(characters) + '"'

        self.value = self.value + 1
        return nextValue


class DateTimeIter(object):

    def __init__(self):
        self.value = 0

    def __str__(self):
        return 'datetime'

    def __iter__(self):
        return self

    def next(self):
        nextValue = self.value
        self.value = self.value + 1
        return numpy.datetime64(nextValue, 'D')


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("Please define number of records in datasets: ")

    numRecords = int(sys.argv[1])

    output = open('./data/ints', 'w')
    generate_dataset(output, IntIter(), ',', numRecords)
    output.close()

    output = open('./data/floats', 'w')
    generate_dataset(output, FloatIter(), ',', numRecords)
    output.close()

    output = open('./data/missingvalues', 'w')
    generate_dataset(output, MissingValuesIter(), ',', numRecords)
    output.close()

    output = open('./data/fixedwidths', 'w')
    generate_dataset(output, FixedWidthIter(), '', numRecords)
    output.close()

    input = open('./data/ints', 'rb')
    output = gzip.open('./data/ints.gz', 'wb')
    output.writelines(input)
    output.close()
    input.close

    '''generate_dataset('ints2', IntIter(), ',', 12500000)
    generate_dataset('ints3', IntIter(), ',', 25000000)
    generate_dataset('signedints1', SignedIntIter(), ',', 2500000)
    generate_dataset('floats1', FloatIter(), ',', 1500000)
    generate_dataset('floats2', FloatIter(), ',', 7500000)
    generate_dataset('floats3', FloatIter(), ',', 15000000)
    generate_dataset('missingvalues1', MissingValuesIter(), ',', 3000000)
    generate_dataset('missingvalues2', MissingValuesIter(), ',', 15000000)
    generate_dataset('missingvalues3', MissingValuesIter(), ',', 30000000)
    generate_dataset('fixedwidth1', FixedWidthIter(), '', 5000000)
    generate_dataset('fixedwidth2', FixedWidthIter(), '', 25000000)
    generate_dataset('fixedwidth3', FixedWidthIter(), '', 50000000)
    generate_dataset('ints_spacedelim', IntIter(), ' ', 2500000)
    generate_dataset('quotes', QuoteIter(), ' ', 2500000)
    generate_dataset('datetime', DateTimeIter(), ',', 2500000)'''

