import pyopencl as cl
import numpy
from PIL import Image

import sys

img = Image.open("../characterRecognition/2.jpg")
img_arr = numpy.asarray(img).astype(numpy.uint8)
dim = img_arr.shape

host_arr = img_arr.reshape(-1)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_arr)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, host_arr.nbytes)

kernel_code = """
    __kernel void copyImage(__global const uint8 *a, __global uint8 *c)
    {
        int rowid = get_global_id(0);
        int colid = get_global_id(1);

        int ncols = %d;
        int npix = %d; //number of pixels, 3 for RGB 4 for RGBA

        int index = rowid * ncols * npix + colid * npix;
        c[index + 0] = a[index + 0];
        c[index + 1] = a[index + 1];
        c[index + 2] = a[index + 2];
    }
    """ % (dim[1], dim[2])

prg = cl.Program(ctx, kernel_code).build()

prg.copyImage(queue, (dim[0], dim[1]), None, a_buf, dest_buf)

result = numpy.empty_like(host_arr)
cl.enqueue_copy(queue, result, dest_buf)

result_reshaped = result.reshape(dim)
img2 = Image.fromarray(result_reshaped, "RGB")
img2.save("new_image_gpu.bmp")
