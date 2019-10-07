#
#Summer Project--Particle Shape Analysis
#
#Import extensive modules
import math
from PIL import Image
def imIn():
    myFile= input("Please enter the file name:")
    num = int(input("Please enter the number of particles in the image:"))
    Num = int(math.sqrt(num))
    print(Num)
    return myFile, Num

def crop():
    #import the image
    myFile, Num =imIn()
    img = Image.open(myFile)
    width, height = img.size
    chopsize = int(img.size[0])//Num
    print(img.size)
    print(chopsize)
    
    # Save Chops of original image
    
    for x0 in range(0, width, chopsize):
       for y0 in range(0, height, chopsize):
          box = (x0, y0,
                 x0+chopsize if x0+chopsize <  width else  width - 1,
                 y0+chopsize if y0+chopsize < height else height - 1)
      
          img.crop(box).save('zchop.%s.x%03d.y%03d.png' % (myFile.replace('.png',''), x0, y0))

#Call the crop function
crop()

##END##

