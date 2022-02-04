import cv2
import os
import matplotlib.pyplot as plt

name=input(f"Enter the image name: ")
image = cv2.imread(f'./uploads/{name}.png')
# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
cv2.imshow('Grayscale', gray_image)
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# creating plot for testing image_
# saving the figure.
os.makedirs("plots", exist_ok=True)
path=os.path.join(f"plots/","{name}.png")
print(path)
plt.savefig(path)
plt.show()
