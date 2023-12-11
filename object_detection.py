import cv2, sys
import numpy as np
import os

dir_img = "/Users/quangtn/Desktop/working_space/job_1/Source/viact/" \
          "mmsegmentation_viact-viact/viact/test/test_data/test_images/" \
          "Ashe_231705051716794_round3_Ashe_06-07-2021.mp4_26_2.jpg"

def find_objects(filename):
    tmp = filename.split(".")
    files = tmp[0].split("/")
    dir_output = "process_image/" + files[len(files)-1]

    names = files[len(files)-1].split(".")

    os.makedirs(dir_output, exist_ok=True)
    image = cv2.imread(filename, 1)
    original_image = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = original_image[y:y + h, x:x + w]
        image_name = dir_output+"/"+names[0]+"_" + str(i + 1) + ".jpg"
        cv2.imwrite(image_name, cropped_contour)


if __name__ == "__main__":
    import glob
    dir_imgs = glob.glob("test_data/test_images/*")
    for i in dir_imgs:
        find_objects(i)
    #find_objects(dir_img)