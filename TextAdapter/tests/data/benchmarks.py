import timeit
import os


def timeFunction(function, setup):
    print 'timing', function
    t = timeit.Timer(stmt=function, setup=setup)
    times = []
    for i in range(0,3):
        os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')
        times.append(str(t.timeit(number=1)))
    return min(times)


ints1 = timeFunction('blazeopt.loadtxt("ints1", dtype="u4,u4,u4,u4,u4", delimiter=",")', 'import blazeopt')
ints2 = timeFunction('blazeopt.loadtxt("ints2", dtype="u4,u4,u4,u4,u4", delimiter=",")', 'import blazeopt')
ints3 = timeFunction('blazeopt.loadtxt("ints3", dtype="u4,u4,u4,u4,u4", delimiter=",")', 'import blazeopt')
print ints1, ints2, ints3

floats1 = timeFunction('blazeopt.loadtxt("floats1", dtype="f8,f8,f8,f8,f8", delimiter=",")', 'import blazeopt')
floats2 = timeFunction('blazeopt.loadtxt("floats2", dtype="f8,f8,f8,f8,f8", delimiter=",")', 'import blazeopt')
floats3 = timeFunction('blazeopt.loadtxt("floats3", dtype="f8,f8,f8,f8,f8", delimiter=",")', 'import blazeopt')
print floats1, floats2, floats3

ints1 = timeFunction('blazeopt.genfromtxt("ints1", dtype="u4,u4,u4,u4,u4", delimiter=",")', 'import blazeopt')
ints2 = timeFunction('blazeopt.genfromtxt("ints2", dtype="u4,u4,u4,u4,u4", delimiter=",")', 'import blazeopt')
ints3 = timeFunction('blazeopt.genfromtxt("ints3", dtype="u4,u4,u4,u4,u4", delimiter=",")', 'import blazeopt')
print ints1, ints2, ints3

floats1 = timeFunction('blazeopt.genfromtxt("floats1", dtype="f8,f8,f8,f8,f8", delimiter=",")', 'import blazeopt')
floats2 = timeFunction('blazeopt.genfromtxt("floats2", dtype="f8,f8,f8,f8,f8", delimiter=",")', 'import blazeopt')
floats3 = timeFunction('blazeopt.genfromtxt("floats3", dtype="f8,f8,f8,f8,f8", delimiter=",")', 'import blazeopt')
print floats1, floats2, floats3

missingValues1 = timeFunction('blazeopt.genfromtxt("missingvalues1", dtype="u4,u4,u4,u4,u4", delimiter=",", missing_values={0:["NA","NaN"], 1:["xx","inf"]}, filling_values="999")', 'import blazeopt')
missingValues2 = timeFunction('blazeopt.genfromtxt("missingvalues2", dtype="u4,u4,u4,u4,u4", delimiter=",", missing_values={0:["NA","NaN"], 1:["xx","inf"]}, filling_values="999")', 'import blazeopt')
missingValues3 = timeFunction('blazeopt.genfromtxt("missingvalues3", dtype="u4,u4,u4,u4,u4", delimiter=",", missing_values={0:["NA","NaN"], 1:["xx","inf"]}, filling_values="999")', 'import blazeopt')
print missingValues1, missingValues2, missingValues3

fixedwidth1 = timeFunction('blazeopt.genfromtxt("fixedwidth1", dtype="u4,u4,u4,u4,u4", delimiter=[2,3,4,5,6])', 'import blazeopt')
fixedwidth2 = timeFunction('blazeopt.genfromtxt("fixedwidth2", dtype="u4,u4,u4,u4,u4", delimiter=[2,3,4,5,6])', 'import blazeopt')
fixedwidth3 = timeFunction('blazeopt.genfromtxt("fixedwidth3", dtype="u4,u4,u4,u4,u4", delimiter=[2,3,4,5,6])', 'import blazeopt')
print fixedwidth1, fixedwidth2, fixedwidth3

