#!/usr/bin/env python3
import subprocess
import sys

# image, crop- dimensions
img = sys.argv[1]; left = sys.argv[2]; right = sys.argv[3]; top = sys.argv[4]; bottom = sys.argv[5]
# arrange the output file's name and path
img_base = img[:img.rfind(".")]; extension = img[img.rfind("."):]; path = img[:img.rfind("/")]
img_out = img_base+"[cropped]"+extension
# get the current img' size
data = subprocess.check_output(["identify", img]).decode("utf-8").strip().replace(img, "")
size = [int(n) for n in data.replace(img, "").split()[1].split("x")]
# calculate the command to resize
w = str(size[0]-int(left)-int(right)); h = str(size[1]-int(top)-int(bottom)); x = left; y = top
# execute the command
cmd = ["convert", img, "-crop", w+"x"+h+"+"+x+"+"+y, "+repage", img_out]
subprocess.Popen(cmd)

