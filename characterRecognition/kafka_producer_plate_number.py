from confluent_kafka import Producer
import socket
import json

conf = {'bootstrap.servers': "localhost:9092",
        'client.id': socket.gethostname()}

producer = Producer(conf)


def acked(err, msg):
    if err is not None:
        print("Failed wto deliver message: %s: %s" % (str(msg.value().decode("utf-8")), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value().decode("utf-8"))))


def sendMess(mess, image):
    data = {'regPlate': mess, 'imgPath': image}
    msg = json.dumps(data)
    producer.produce("registration_plate", key="key", value=msg, callback=acked)
    producer.poll(1)
    producer.flush(2)
    print(data)
