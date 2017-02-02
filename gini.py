import sys, json, numpy
prods = numpy.cumsum(sorted(numpy.array(json.loads(open(sys.argv[1], 'r').read())['productions']).flatten()))

production = numpy.array(json.loads(open(sys.argv[1], 'r').read())['productions'])
frames = numpy.array(json.loads(open(sys.argv[1], "r").read())['frames'])

#prods = numpy.cumsum(sorted(numpy.array(json.loads(open("1485967728-3533319074.hlt", 'r').read())['productions']).flatten()))

#production = numpy.array(json.loads(open("1485967728-3533319074.hlt", 'r').read())['productions'])
#frames = numpy.array(json.loads(open("1485967728-3533319074.hlt", "r").read())['frames'])
str = frames[1, :, :, 1]

vals = numpy.maximum(production, 0.1) / numpy.maximum(str, 1)
vals2 = numpy.cumsum(sorted(vals.flatten()))

rec = numpy.maximum(str, 1) / numpy.maximum(production, .1)
rec2 = numpy.cumsum(sorted(rec.flatten()))

# print(str((len(prods)*prods[-1]-2*numpy.trapz(prods)+prods[0])/len(prods)/prods[-1]))
print((len(prods)*prods[-1]-2*numpy.trapz(prods)+prods[0])/len(prods)/prods[-1])
# print(str((len(vals2)*vals2[-1]-2*numpy.trapz(vals2)+vals2[0])/len(vals2)/vals2[-1]))
print((len(vals2)*vals2[-1]-2*numpy.trapz(vals2)+vals2[0])/len(vals2)/vals2[-1])

print((len(rec2)*rec2[-1]-2*numpy.trapz(rec2)+rec2[0])/len(rec2)/rec2[-1])
