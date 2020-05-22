class ExampleHandler(FileSystemEventHandler):
    def on_created(self, event):  # when file is created
        # do something, eg. call your function to process the image
        pathToWoodVideo = event.src_path
        # time.sleep(15)
        print("Got event for file %s" % pathToWoodVideo)
        t = threading.Thread(target=startPlateDetection(pathToWoodVideo),
                             name="startingPlateDetection")
        t.daemon = True
        t.start()
        # startPlateDetection(pathToWoodVideo)


observer = Observer()
event_handler = ExampleHandler()  # create event handler
# set observer to use created handler in directory
observer.schedule(event_handler, path='detectedWood')
observer.start()

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
