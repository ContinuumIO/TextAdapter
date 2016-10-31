import sys, os
import iopro.pyodbc as pyodbc
import numpy as np
from time import time
import subprocess


_default_connect_string = r'''Driver={SQL Server};SERVER=WIN-G92S2JK4KII;DATABASE=Test;USER=Development;PWD=C0ntinuum'''
        #'DSN=myodbc3;UID=faltet;PWD=continuum;DATABASE=test')
        #'DSN=PSQL;UID=faltet;PWD=continuum;DATABASE=test')
        #'DSN=odbcsqlite;DATABASE=market.sqlite')
        #'DSN=odbcsqlite;DATABASE=market-1k.sqlite')

# The number of rows in the table
#N = 1*1000
N = 1000*1000
# Whether we want a mem profile or not
profile = False

def show_stats(explain, tref):
    "Show the used memory (only works for Linux 2.6.x)."
    # Build the command to obtain memory info
    cmd = "cat /proc/%s/status" % os.getpid()
    sout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    for line in sout:
        if line.startswith("VmSize:"):
            vmsize = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            vmrss = int(line.split()[1])
        elif line.startswith("VmData:"):
            vmdata = int(line.split()[1])
        elif line.startswith("VmStk:"):
            vmstk = int(line.split()[1])
        elif line.startswith("VmExe:"):
            vmexe = int(line.split()[1])
        elif line.startswith("VmLib:"):
            vmlib = int(line.split()[1])
    sout.close()
    print "Memory usage: ******* %s *******" % explain
    print "VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss)
    print "VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk)
    print "VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib)
    tnow = time()
    print "WallClock time:", round(tnow - tref, 3)
    return tnow


def write(cursor):

    try:
        cursor.execute('drop table market')
    except:
        pass
    cursor.execute('create table market '
                   '(symbol_ varchar(5),'
                   ' open_ float, low_ float, high_ float, close_ float,'
                   ' volume_ int)')
    # cursor.execute('create table market '
    #                '(symbol varchar(5) not null,'
    #                ' open float not null, low float not null, high float not null, close float not null,'
    #                ' volume int not null)')

    # Add content to 'column' row by row
    t0 = time()
    for i in xrange(N):
        cursor.execute(
            "insert into market(symbol_, open_, low_, high_, close_, volume_)"
            " values (?, ?, ?, ?, ?, ?)",
            #(str(i), float(i), float(2*i), float(3*i), float(4*i), i))
            (str(i), float(i), float(2*i), None, float(4*i), i))
    cursor.execute("commit")             # not supported by SQLite
    t1 = time() - t0
    print "Stored %d rows in %.3fs" % (N, t1)

def read(cursor, fields):

    if fields == "full":
        query = "select * from market"
        dtype = "S5,f4,f4,f4,f4,i4"
    else:
        query = "select volume from market"
        dtype = "i4"

    # cursor.execute(query)
    # t0 = time()
    # if fields == "full":
    #     siter = np.fromiter((
    #         (row[0], row[1], row[2], row[3], row[4], row[5])
    #         for row in cursor), dtype=dtype)
    # else:
    #     siter = np.fromiter((row[0] for row in cursor), dtype=dtype)
    # t1 = time() - t0
    # print "[cursoriter] Retrieved %d rows in %.3fs" % (len(siter), t1)
    # print "Last row:", siter[-1]

    # t0 = time()
    # cursor.execute(query)
    # t1 = time() - t0
    # print "[execute query] Query executed in %.3fs" % (t1,)
    # t0 = time()
    # sarray = cursor.fetchsarray()
    # t1 = time() - t0
    # print "[fetchsarray] Retrieved %d rows in %.3fs" % (len(sarray), t1)
    # print "Last row:", sarray[-1], sarray.dtype
    # del sarray
#    cursor = c.cursor()
    t0 = time()
    cursor.execute(query)
    t1 = time() - t0
    print "[execute query] Query executed in %.3fs" % (t1,)
    if profile: tref = time()
    if profile: show_stats("Before reading...", tref)
    t0 = time()
    all = cursor.fetchall()
    t1 = time() - t0
    if profile: show_stats("After reading...", tref)
    print "[fetchall] Retrieved %d rows in %.3fs" % (len(all), t1)
    print "Last row:", all[-1]
    del all
#    del cursor
    time1 = t1

#    cursor = c.cursor()
    t0 = time()
    cursor.execute(query)
    t1 = time() - t0
    print "[execute query] Query executed in %.3fs" % (t1,)
    if profile: tref = time()
    if profile: show_stats("Before reading...", tref)
    t0 = time()
    dictarray = cursor.fetchdictarray(N)
    t1 = time() - t0
    if profile: show_stats("After reading...", tref)
    print "[fetchdictarray(N)] Retrieved %d rows in %.3fs" % (len(dictarray['volume_']), t1)
    print "Last row:", [(name, arr[-1]) for name, arr in dictarray.iteritems()]
    del dictarray
#    del cursor
    time2 = t1

#    cursor = c.cursor()
    t0 = time()
    cursor.execute(query)
    t1 = time() - t0
    print "[execute query] Query executed in %.3fs" % (t1,)
    if profile: tref = time()
    if profile: show_stats("Before reading...", tref)
    t0 = time()
    dictarray = cursor.fetchdictarray()
    t1 = time() - t0
    if profile: show_stats("After reading...", tref)
    print "[fetchdictarray] Retrieved %d rows in %.3fs" % (len(dictarray['volume_']), t1)
    print "Last row:", [(name, arr[-1]) for name, arr in dictarray.iteritems()]
    del dictarray
#    del cursor
    time3 = t1

#    cursor = c.cursor()
    t0 = time()
    cursor.execute(query)
    t1 = time() - t0
    print "[execute query] Query executed in %.3fs" % (t1,)
    if profile: tref = time()
    if profile: show_stats("Before reading...", tref)
    t0 = time()
    steps = 1000
    for i in range(steps):
        dictarray = cursor.fetchdictarray(N//steps)
    t1 = time() - t0
    if profile: show_stats("After reading...", tref)
    print "[fetchdictarray twice] Retrieved %d rows in %.3fs" % (len(dictarray['volume_']), t1)
    print "Last row:", [(name, arr[-1]) for name, arr in dictarray.iteritems()]
    del dictarray
#    del cursor
    time4 = t1


    return (time1, time2, time3, time4)


if __name__ == "__main__":

    # set up a connection
    connection = pyodbc.connect(_default_connect_string)
    cursor = connection.cursor()
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "write":
        write(cursor)
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        if sys.platform.startswith("linux"):
            profile = True
        else:
            print "Memory profiling only support on Linux. Exiting..."
            sys.exit(1)

    results = []
    for i in range(5):
        print "\n\nrun %d\n" % i
        results.append(read(cursor, 'full'))
        #read(cursor, 'col')

    for i in range(len(results[0])):
        print np.min([el[i] for el in results])

    connection.close()

