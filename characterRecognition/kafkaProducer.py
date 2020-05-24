from time import sleep
from json import dumps
from kafka import KafkaProducer
import json

#producer = KafkaProducer(value_serializer=lambda m: json.dumps(m).encode('ascii'))
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'))
#future = producer.send('my-topic', b'raw_bytes')
#producer = KafkaProducer(value_serializer=msgpack.dumps)

def sendMess(mess, image):
    data = {'regPlate': mess, 'imgPath': image}
    msg = json.dumps(data)

    producer.send('registration_plate', key="key", value=msg)
    # producer.produce("registration_plate", key="key", value=data, callback=acked)
    #producer.poll(1)
    print(data)

sendMess("waiii", "imagPathBlea")