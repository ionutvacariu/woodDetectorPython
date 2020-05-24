import sys

from confluent_kafka import Consumer
from confluent_kafka.cimpl import KafkaError, KafkaException

conf = {'bootstrap.servers': "localhost:9092",
        'group.id': "registration_plate",
        'auto.offset.reset': 'smallest'}

consumer = Consumer(conf)

running = True


def msg_process(msg):
    print("key" + msg.key().decode("utf-8") + "value : " + msg.value().decode("utf-8"))


def basic_consume_loop(consumer, topics):
    try:
        consumer.subscribe(topics)

        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None: continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                msg_process(msg)
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()


basic_consume_loop(consumer, ["registration_plate"])


def shutdown():
    running = False
