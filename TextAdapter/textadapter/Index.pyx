import numpy
import os
from six import StringIO


cdef void indexer_callback(void *index, uint64_t record_num, uint64_t record_offset):
    """
    Callback function for C parsing engine to store record number
    and offset in ExactIndex object.
    """

    idx = <object>index
    idx.set_offset(record_num, record_offset)


cdef RecordOffset index_lookup_callback(void *index, uint64_t record_num):
    """
    Callback function for C parsing engine to retrieve offset
    for record number. get_offset returns the offset of the record
    in the index that is closest to, but not greater than, the
    requested one.
    """
    idx = <object>index
    cdef RecordOffset offset
    offset.record_num, offset.offset = idx.get_offset(record_num)
    return offset


cdef void add_gzip_access_point_callback(void *index, unsigned char *window, uint32_t compressed_offset, uint64_t uncompressed_offset,
    int avail_in, int avail_out, uint8_t bits):

    idx = <object>index
    idx.add_gzip_access_point(window, compressed_offset, uncompressed_offset, avail_in, avail_out, bits)


cdef void get_gzip_access_point_callback(void *index, uint64_t offset, GzipIndexAccessPoint *point):

    idx = <object>index
    bits, compressed_offset, uncompressed_offset, window = idx.get_gzip_access_point(offset)
    point.bits = bits
    point.compressed_offset = compressed_offset
    point.uncompressed_offset = uncompressed_offset
    memcpy(<char*>point.window, <char*>window, UNCOMPRESSED_WINDOW_SIZE)


class ExactIndex(object):
    """Implementation of an exact index on-disk.

    The following ASCII representation shows the layout of the header::

        |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
        | t   a   i   x | ^ | ^ | ^ | ^ |    density    |    RESERVED   |
                          |   |   |   |
              version ----+   |   |   |
              options --------+   |   |
<<<<<<< HEAD:iopro/textadapter/Index.pyx
<<<<<<< HEAD
             checksum ***REMOVED***-+   |
             typesize ***REMOVED***-----+
=======
             checksum ------------+   |
             typesize ----------------+
>>>>>>> 14dcbb9542f8d05344fd4a2cc4ef07c47528a8f1
=======
             checksum ------------+   |
             typesize ----------------+
>>>>>>> 0e94e8123ce07aa964a82f678b115c7defb0a49c:TextAdapter/textadapter/Index.pyx

        |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
        |          num_offsets          |    num_gzip_access_points     |

        |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
        |       total_num_records       |            RESERVED           |

        |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
        |           RESERVED            |            RESERVED           |


    The 'taix' magic bytes stand for 'Text Adapter IndeX'.  See below for
    other fields.

    Description of the header entries
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    All entries are little-endian.

    :version:
        (``uint8``)
        Format version of the TAIX header, to ensure exceptions in case of
        forward incompatibilities.

    :options:
        (``bitfield``)
        A bitfield which allows for setting certain options in this file.

        :``bit 0 (0x01)``:
            If the offsets in this file are compressed.

    :checksum:
        (``uint8``)
        The checksum used. The following checksums, available in the python
        standard library should be supported. The checksum is always computed
        on the compressed data and placed after the chunk.

        :``0``:
            ``no checksum``
        :``1``:
            ``zlib.adler32``
        :``2``:
            ``zlib.crc32``
        :``3``:
            ``hashlib.md5``
        :``4``:
            ``hashlib.sha1``
        :``5``:
            ``hashlib.sha224``
        :``6``:
            ``hashlib.sha256``
        :``7``:
            ``hashlib.sha384``
        :``8``:
            ``Hashlib.sha512``

    :typesize:
        (``uint8``)
        The typesize of the data in the index. Currently, assume that the
        default typesize is 8 (uint64).

    :density:
        (``uint32``)
        Denotes the density of the index.  A value of 1 means a fully dense
        index (i.e. 1 offset entry per record), while a value N larger than 1
        means and sparse index (i.e. 1 offset entry per N records).

    :num_offsets:
        (``uint64``)
        Denotes the number of records in the text file. Value of zero means
        no records have been indexed.
    """

    TAG_OFFSET = 0
    TAG_SIZE = 4

    VERSION_OFFSET = 4
    VERSION_SIZE = 1
    OPTIONS_OFFSET = 5
    OPTIONS_SIZE = 1
    CHECKSUM_OFFSET = 6
    CHECKSUM_SIZE = 1
    TYPESIZE_OFFSET = 7
    TYPESIZE_SIZE = 1

    DENSITY_OFFSET = 8
    DENSITY_OFFSET = 4

    RESERVED1_OFFSET = 12
    RESERVED1_SIZE = 4

    NUMOFFSETS_OFFSET = 16
    NUMOFFSETS_SIZE = 8

    NUM_GZIP_ACCESS_POINTS_OFFSET = 24
    NUM_GZIP_ACCESS_POINTS_SIZE = 8

    TOTAL_NUM_RECORDS_OFFSET = 32
    TOTAL_NUM_RECORDS_SIZE = 8

    HEADER_SIZE = 64

    def __init__(self, index_name=None, density=DEFAULT_INDEX_DENSITY, typesize=8, num_records=0):
        if density > 2**16-1:
            raise ValueError("density is too large")
        self.density = density
        self.typesize = typesize
        self.num_offsets = 0
        self.num_gzip_access_points = 0
        self.total_num_records = num_records
        global boto_installed

        if index_name is None:
            self.offsets = []
            self.gzip_access_points = []
            self.idxfile = None
        else:
            if isinstance(index_name, basestring):
                self.idxfile = index_name
                if not os.path.exists(self.idxfile):
                    self.create_disk_index()
                else:
                    self.open_disk_index()
            elif boto_installed and isinstance(index_name, key.Key):
                
                self.idxfile = key.Key(index_name.bucket)
                self.idxfile.key = index_name.key + '.idx'
                self.open_s3_index()


    def create_disk_index(self):
        """Create an exact index following the format specifcation."""

        # The dtype for the offsets
        self.dtype = numpy.dtype("u%d"%self.typesize)

        # Write index header for version 1 of the format
        idxh = open(self.idxfile, 'wb')
        idxh.write(b"taix")  # the 'magic' header
        bytes = numpy.empty(4, dtype='u1')
        bytes[0] = 0x1   # the version
        bytes[1] = 0x0   # the options
        bytes[2] = 0x0   # the checksum method
        bytes[3] = self.typesize   # the typesize for offsets
        bytes.tofile(idxh)
        # The density of the index
        numpy.array(self.density, dtype='u4').tofile(idxh)
        # Now, a reserved field
        numpy.array(-1, dtype='i4').tofile(idxh)
        # The number of offsets is still unknown
        numpy.array(0, dtype='u8').tofile(idxh)
        # The number of gzip access points is still unknown
        numpy.array(0, dtype='u8').tofile(idxh)
        # The total number of records in file
        numpy.array(self.total_num_records, dtype='u8').tofile(idxh)
        # Reserved field
        numpy.array(-1, dtype='i8').tofile(idxh)
        # Reserved field
        numpy.array(-1, dtype='i8').tofile(idxh)
        # Reserved field
        numpy.array(-1, dtype='i8').tofile(idxh)
        self.header_length = self.HEADER_SIZE
        self.idxh = idxh

        self.idxh.close()
        self.idxh = open(self.idxfile, 'r+b')


    def open_disk_index(self):
        """Open an existing index"""
        self.idxh = idxh = open(self.idxfile, 'r+b')
        # Read header.
        magic = idxh.read(4)
        if magic != b"taix":
            raise ValueError("The magic value '%s' is not recognized!" % magic)
        version, options, checksum, typesize = numpy.fromstring(
            idxh.read(4), dtype='u1')
        if version == 1:
            self.header_length = self.HEADER_SIZE
        self.typesize = int(typesize)
        self.dtype = numpy.dtype("u%d"%self.typesize)
        # The density
        self.density = long(numpy.fromfile(idxh, dtype='u4', count=1)[0])
        # A reserved field
        idxh.read(4)
        # The number of offsets
        num_offsets = numpy.fromfile(idxh, dtype='u8', count=1)[0]
        self.num_offsets = long(num_offsets)

        num_gzip_access_points = numpy.fromfile(idxh, dtype='u8', count=1)[0]
        self.num_gzip_access_points = long(num_gzip_access_points)

        total_num_records = numpy.fromfile(idxh, dtype='u8', count=1)[0]
        self.total_num_records = long(total_num_records)


    def open_s3_index(self):
        """Open an existing S3 index"""
        data = StringIO(self.idxfile.get_contents_as_string(headers={'Range' : 'bytes={0}-{1}'.format(0, self.HEADER_SIZE)}))

        # Read header.
        magic = data.read(4)
        if magic != "taix":
            raise ValueError("The magic value '%s' is not recognized!" % magic)
        version, options, checksum, typesize = numpy.fromstring(
            data.read(4), dtype='u1')
        if version == 1:
            self.header_length = self.HEADER_SIZE
        self.typesize = int(typesize)
        self.dtype = numpy.dtype("u%d"%self.typesize)
        # The density
        self.density = long(numpy.fromstring(data.read(4), dtype='u4'))
        # A reserved field
        data.read(4)
        # The number of offsets
        num_offsets = numpy.fromstring(data.read(8), dtype='u8')
        self.num_offsets = long(num_offsets)

        num_gzip_access_points = numpy.fromstring(data.read(8), dtype='u8')
        self.num_gzip_access_points = long(num_gzip_access_points)

        total_num_records = numpy.fromstring(data.read(8), dtype='u8')
        self.total_num_records = long(total_num_records)


    def get_offset(self, rec_num):
        """Get the offset for `rec_num`."""
        global boto_installed

        if self.num_offsets == 0:
            return (0, 0)

        # Do a seek in index file depending on the sign of the offset
        if rec_num < 0:
            raise ValueError('Invalid record number')

        ioffset, num_offsets = divmod(rec_num, self.density)
        if not -self.num_offsets <= ioffset < self.num_offsets:
            #raise IndexError("index out of range")
            ioffset = self.num_offsets - 1

        if self.idxfile is None:
            return (ioffset * self.density, self.offsets[ioffset])
        elif isinstance(self.idxfile, basestring):
            pos = ioffset * self.typesize + self.header_length
            self.idxh.seek(pos, os.SEEK_SET)
            offset = numpy.fromfile(self.idxh, dtype=self.dtype, count=1)[0]
            return (ioffset * self.density,offset)
        elif boto_installed and isinstance(self.idxfile, key.Key):
            pos = ioffset * self.typesize + self.header_length
            data = StringIO(self.idxfile.get_contents_as_string(headers={'Range' : 'bytes={0}-{1}'.format(pos, pos+self.typesize)}))
            offset = long(numpy.fromstring(data.read(8), dtype='u8'))
            return (ioffset * self.density, offset)


    def set_offset(self, rec_num, offset):
        """
        Set offset for rec_num.
        Offset is added at end of offsets array.
        """

        # Check to see if we've already indexed this record
        ioffset, num_offsets = divmod(rec_num, self.density)
        if ioffset < self.num_offsets:
            return
       
        if self.idxfile is None:
            self.offsets.append(offset)
        elif isinstance(self.idxfile, basestring):
            self.idxh.seek(0, 2)
            numpy.array(offset, dtype=self.dtype).tofile(self.idxh)

        # Increment number of records in header
        self.num_offsets += 1

    def add_gzip_access_point(self, window, compressed_offset, uncompressed_offset, avail_in, avail_out, bits):

        if self.idxfile is None:        
            self.gzip_access_points.append((bits, compressed_offset, uncompressed_offset, window))
        elif isinstance(self.idxfile, basestring):
            self.idxh.seek(0, 2)
            numpy.array(bits, dtype='u1').tofile(self.idxh)
            numpy.array(compressed_offset, dtype='u8').tofile(self.idxh)
            numpy.array(uncompressed_offset, dtype='u8').tofile(self.idxh)
            numpy.array(window, dtype='S32768').tofile(self.idxh)

        self.num_gzip_access_points += 1


    def get_gzip_access_point(self, offset):

        if self.idxfile is None:
            for i, access_point in enumerate(self.gzip_access_points):
                if offset == access_point[2]:
                    return access_point
                elif offset < access_point[2] and i > 0:
                    return self.gzip_access_points[i-1]
            return self.gzip_access_points[-1]
        elif isinstance(self.idxfile, basestring):
            pos = self.num_offsets * self.typesize + self.header_length
            self.idxh.seek(pos, 0)

            prev_bits = 0
            prev_compressed_offset = 0
            prev_uncompressed_offset = 0
            prev_window = ''

            for i in range(self.num_gzip_access_points):
                bits = numpy.fromfile(self.idxh, dtype='u1', count=1)[0]
                compressed_offset = numpy.fromfile(self.idxh, dtype='u8', count=1)[0]
                uncompressed_offset = numpy.fromfile(self.idxh, dtype='u8', count=1)[0]
                window = numpy.fromfile(self.idxh, dtype='S32768', count=1)[0]
                if offset == uncompressed_offset:
                    return (bits, compressed_offset, uncompressed_offset, window)
                elif offset < uncompressed_offset and i > 0:
                    return (prev_bits, prev_compressed_offset, prev_uncompressed_offset, prev_window)
                prev_bits = bits
                prev_compressed_offset = compressed_offset
                prev_uncompressed_offset = uncompressed_offset
                prev_window = window
            return (prev_bits, prev_compressed_offset, prev_uncompressed_offset, prev_window)


    def finalize(self, total_num_records):
        """
        This function is called by C parsing engine when parsing is finished.
        Any post-indexing stuff should be put here.
        """

        self.total_num_records = total_num_records

        if isinstance(self.idxfile, basestring):
            # Save number of offsets and gzip access_points in file
            pos = self.idxh.tell()

            self.idxh.seek(self.NUMOFFSETS_OFFSET, os.SEEK_SET)
            numpy.array(self.num_offsets, dtype='u8').tofile(self.idxh)
            self.idxh.seek(self.NUM_GZIP_ACCESS_POINTS_OFFSET, os.SEEK_SET)
            numpy.array(self.num_gzip_access_points, dtype='u8').tofile(self.idxh)
            self.idxh.seek(self.TOTAL_NUM_RECORDS_OFFSET, os.SEEK_SET)
            numpy.array(total_num_records, dtype='u8').tofile(self.idxh)

            self.idxh.seek(pos, os.SEEK_SET)


    def __len__(self):
        return self.num_offsets

    def get_density(self):
        return self.density

    def close(self):
        if isinstance(self.idxfile, basestring):
            self.idxh.close()


# The chunksize that is read by default
CS = 128 * 1024   # 128 KB

class FuzzyIdx(object):
    """Implementation of a fuzzy index"""

    def __init__(self, textfileh, skip):
        self.textfileh = textfileh
        self.skip = skip
        # The filesize and the number of bytes to skip
        self.filesize, self.skipbytes = self.get_size_skipbytes(skip)

        # Get a sensible chunksize for the retrieved blocks as well as the
        # inital sizes
        self.chunksize, isizes = self.get_sample_chunksize()
        # Get values at the end
        esizes = self.chunk_sizes(where=-1, chunksize=self.chunksize)
        self.isizes, self.esizes = isizes, esizes[::-1]
        self.imean, self.emean = isizes.mean(), esizes.mean()
        self.istd, self.estd = isizes.std(), esizes.std()
        #print "means:", self.imean, self.emean

        # Compute mean for all values
        allsizes = numpy.concatenate((isizes, esizes))
        self.std = allsizes.std()
        self.mean = allsizes.mean()
        # Shrink the mean by the stddev.  This allows for a larger estimation
        # of the number of records, which can be beneficial (i.e. requires
        # less enlargements) when creating a big recarray for keeping all the
        # data.
        shmean = self.mean - self.std
        self.nrecords = int((self.filesize - self.skipbytes) / shmean)
        #print "skipbytes:", self.skipbytes
        #print "mean:", self.mean
        #print "stddevs:", self.istd, self.estd
        #print "nrecords:", self.nrecords

    def chunk_sizes(self, where=0, skipbytes=0, chunksize=CS):
        """Return the sizes for records in a chunk of text."""
        if where == 0:
            self.textfileh.seek(skipbytes, 0)
            chunk = self.textfileh.read(chunksize)
            chunk = chunk[:chunk.rfind('\n')]
        elif where == -1:
            self.textfileh.seek(-chunksize, 2)
            chunk = self.textfileh.read(chunksize)
            chunk = chunk[chunk.find('\n')+1:]
        else:
            raise NotImplementedError
        start = 0; sizes = []
        while True:
            end = chunk.find('\n', start)
            if end < 0: break
            sizes.append(end-start+1)  # probably +2 for '\r\n' (Win)
            start = end+1
        asizes = numpy.array(sizes, dtype='i4')
        return asizes

    def get_sample_chunksize(self):
        isizes = numpy.empty(0, dtype='i4')
        chunksize = CS; skipbytes = self.skipbytes
        asizes = self.chunk_sizes(
        where=0, skipbytes=skipbytes, chunksize=chunksize)
        isizes = numpy.concatenate((isizes, asizes))
        return chunksize, isizes

    def align_fuzzy(self, offset):
        """Aling an offset so that it starts after a newline."""
        # The number of bytes to read (should be a multiple of 2)
        NB = int(self.mean * 10)     # Read a 10 lines window
        hnb = NB / 2

        # Position at the beginning of the NB window
        if abs(offset) < hnb:
            hnb = abs(offset)
        if offset >= 0:
            self.textfileh.seek(offset - hnb, os.SEEK_SET)
        else:
            self.textfileh.seek(offset - hnb, os.SEEK_END)
        # Read the window
        chunk = self.textfileh.read(NB)

        # Start finding the '\\n' from the midpoint to backwards
        bpos = hnb - chunk[:hnb].rfind('\n')
        if bpos > hnb:
            raise IOError("\\n not detected in the backward direction!")
        # Start finding the '\n' from the midpoint to forward
        fpos = chunk[hnb-1:].find('\n')
        if fpos < 0:
            raise IOError("\\n not detected in the forward direction!")
        # Select the position nearest to the midpoint
        if bpos < fpos:
            # \n found before the midpoint
            pos = hnb - bpos + 1
        else:
            # \n found after the midpoint
            pos = hnb + fpos

        return offset - hnb + pos

    def seek_offset(self, rec_num):
        """Get the offset for `rec_num` *and* positionate there."""
        if not -self.nrecords <= rec_num < self.nrecords:
            raise IndexError("index out ot range")

        if rec_num > (self.nrecords / 2):
            rec_num = -(self.nrecords - rec_num)

        fuzzy = False
        # Compute the bytes to skip
        if rec_num >= 0:
            if rec_num < len(self.isizes):
                # Exact index
                offset = self.isizes[:rec_num].sum() + self.skipbytes
            else:
                # Fuzzy index
                offset = int(round(self.imean * rec_num + self.skipbytes))
                fuzzy = True
        else:
            # Complementary rec num
            inv_rec_num = -(rec_num + 1)
            if inv_rec_num < len(self.esizes):
                # Exact index
                offset = -self.esizes[:inv_rec_num+1].sum()
            else:
                # Fuzzy index
                offset = int(round(self.emean * rec_num))
                fuzzy = True

        if fuzzy:
            # Properly align the fuzzy offset
            offset = self.align_fuzzy(offset)

        # Do a seek in file depending on the sign of the offset
        if offset >= 0:
            self.textfileh.seek(offset, os.SEEK_SET)
        else:
            self.textfileh.seek(offset, os.SEEK_END)

    def __len__(self):
        return self.nrecords

    def close(self):
        pass



