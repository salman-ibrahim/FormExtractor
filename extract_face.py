# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:16:14 2020

@author: Salman Malik
"""

import cv2
import os
import datetime
import time
import shutil

#get the starting time to calculate total execution time at the end
start_time = time.time()


#Array to store All Valied image file names

images = []

# Find and add all files in directory and append names to the array of files
files = [f for f in os.listdir(os.curdir) if os.path.isfile(f)]

#filter all the files that are images
for file in files:
    if '.jpg' in file:
        images.append(file)
    elif '.jpeg' in file:
        images.append(file)
    elif '.png' in file:
        images.append(file)
    
            
#Creating a Directory Based on current date and time for every batch scanned to keep record
print ("[INFO] Making directories")
now = datetime.datetime.now()
newDirName = "Result_"+now.strftime("%Y_%m_%d-%H%M")
successDirName = "Success_"+now.strftime("%Y_%m_%d-%H%M")
os.mkdir(newDirName)
os.mkdir(successDirName)

#read all filtered image files one-by-one and extract faces
for f in images:            
    print("[FILE] Current File: "+f)
    image = cv2.imread(f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.8,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    print("[INFO] Found {0} Faces.".format(len(faces)))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        #cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
        print ("[INFO] Writing to path")
        cv2.imwrite(newDirName+"/"+f, roi_color)
        print ("[SUCCESS] Saved face to path " + newDirName)
        print ("[SUCCESS] Moving File to Success Directory")
        
        #if two faces are detected in an image the file will already be moved
        # and it may cause an exception as well as disturb the normal flow of
        # our program so we use try to ignore the exception since it is not a problem
        try:
            shutil.move(f,successDirName+'/'+f)
        except:
            print ("[EXCEPTION] Ignoring the second face in already searched image")
            
print("[STATUS] Execution Time: --- %s seconds ---" % (time.time() - start_time))
print ("[DONE] Task Completed Succesfully")