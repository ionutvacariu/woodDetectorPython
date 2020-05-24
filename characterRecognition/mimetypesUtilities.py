import mimetypes


def get_extensions_for_type(general_type):
    for ext in mimetypes.types_map:
        if mimetypes.types_map[ext].split('/')[0] == general_type:
            yield ext


IMAGE = tuple(get_extensions_for_type('image'))
#print("IMAGE = " + str(IMAGE))
VIDEO = tuple(get_extensions_for_type('video'))
#print("VIDEO = " + str(VIDEO))
