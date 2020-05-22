def substring_after(s, delim):
    return s.partition(delim)[2]


def substring_before(s, delim):
    return s.partition(delim)[0]


def substractVideoName(video_name):
    first = substring_after(video_name, "detectedWood/")
    second = substring_before(first, ".avi")
    return second


video_name = "detectedWood/cutout_wood_withTime1590170606.0705101.avi"
print(substractVideoName(video_name))
