# -*- coding: utf-8 -*-
import pyodbc
from time import time
import sys

# Number of iterations for leak discovery
N = 10000
# Print a profile every M iterations
M = 1000
# The number of rows in table
NR = 100
# Whether we want a mem profile or not
profile = False

def show_stats(explain, tref):
    import os, subprocess
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


def check(cursor):

    print "********************************************"
    cursor.execute("create table t(vc varchar(10), ts timestamp, t time, d date, i int)")
    for i in range(NR):
        input = ("test", "2012-01-01", "00:01:02", "2012-01-01", 0)
        cursor.execute("insert into t values (?,?,?,?,?)", input)
        input = (None, None, None, None, None)
        cursor.execute("insert into t values (?,?,?,?,?)", input)
    if profile: tref = time()
    if profile: show_stats("Before reading...", tref)
    for i in range(N):
        if profile and not i % M: show_stats("After reading...%d"%i, tref)
        result = cursor.execute("select * from t").fetchsarray()
        #result = cursor.execute("select * from t").fetchdictarray()
    if profile: show_stats("After all the process", tref)
    d = result['d']
    #print "shape:", d.shape
    #print "dtype:", d.dtype
    #print "result->", result


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        profile = True

    # set up a connection
    connection = pyodbc.connect(
        #'DSN=myodbc3;UID=faltet;PWD=continuum;DATABASE=test')
        #'DSN=PSQL;UID=faltet;PWD=continuum;DATABASE=test')
        #'DSN=SQLite;DATABASE=market.sqlite')
        'DSN=odbcsqlite;DATABASE=sqlite.db')
        #'DSN=SQLite;DATABASE=market-1k.sqlite')
    cursor = connection.cursor()
    try:
        cursor.execute("drop table t")
    except:
        pass

    check(cursor)
