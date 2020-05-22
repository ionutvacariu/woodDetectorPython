import darknet as dn

#dn.set_gpu(0)

net = dn.load_net("plate_detection_final.weights", "plate_recognition.cfg")
meta = dn.load_meta("obj.data")

r = dn.detect(net, meta, 'object-detection.jpg')
print(r)
