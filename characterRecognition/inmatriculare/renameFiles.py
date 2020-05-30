import os

path = '/Users/ionutvacariu/Downloads/1111/inmatriculare/'
i = 1000
with os.scandir(path) as it:
    for entry in it:
        if entry.is_file() and "PNG" in entry.name:
            file = path + entry.name
            #filename, file_extension = os.path.splitext(file)
            newPath = file.replace('PNG', 'jpg')
            i = i+1
            #ac = file.split('/')[ac.__len__() - 1]
            #nn = os.path.basename(filename)
            # os.rename(nn, str( i + 10))
            os.rename(file, newPath)
            print(newPath)
