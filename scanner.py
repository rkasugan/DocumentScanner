import numpy as np
import cv2
import argparse
import imutils

# Takes list of four points and orders them by top left, top right, bottom right, bottom left
def order_points(in_pts):
    ordered_pts = np.zeros((4,2), dtype="float32")

    sums = np.sum(in_pts, axis = 1)
    diffs = np.diff(in_pts, axis = 1)

    # Points with smallest and largest (x+y) sums are top left and bottom right, respectively
    ordered_pts[0] = in_pts[np.argmin(sums)]
    ordered_pts[2] = in_pts[np.argmax(sums)]

    # Points with smallest and largest (y-x) differences are top right and bottom left, respectively (np.diff does y-x)
    ordered_pts[1] = in_pts[np.argmin(diffs)]
    ordered_pts[3] = in_pts[np.argmax(diffs)]

    return ordered_pts

# Does a linear transform based on four input points
def four_corner_transform(img, in_pts):
    ordered_pts = order_points(in_pts)
    (tl, tr, br, bl) = ordered_pts

    # Calculate largest width and largest height for transform
    width1 = np.sqrt((tl[0]-tr[0])**2 + (tl[1]-tr[1])**2)
    width2 = np.sqrt((bl[0]-br[0])**2 + (bl[1]-br[1])**2)
    width = int(max(width1, width2))

    height1 = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
    height2 = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
    height = int(max(height1, height2))

    # Create target set of points to transform to
    target_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], np.float32)

    # Retrieve the transformation matrix and apply to image
    transform_matrix = cv2.getPerspectiveTransform(ordered_pts, target_pts)
    transformed_img = cv2.warpPerspective(img, transform_matrix, (width,height))
    return transformed_img


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to image")
ap.add_argument("-d", "--debug", required = False, action = "store_true", help = "Output debug pics")
ap.add_argument("-n", "--name", required = False, help = "Name of output file")
args = vars(ap.parse_args())

img_to_transform = cv2.imread(args["image"])
orig = img_to_transform.copy()

ratio = img_to_transform.shape[0] / 500.0
img_to_transform = imutils.resize(img_to_transform, height = 500)

# canny edge detection (with necessary removal of noise)
gray = cv2.cvtColor(img_to_transform, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = imutils.auto_canny(gray)  # cv2.Canny would be more precise, but a scan is pretty simple so should work fine
if args["debug"]:
    cv2.imwrite("Edged.png", edged)

# find contours
contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours) # handles the formatting of numbers (float32 and stuff)
contours = sorted(contours, key = cv2.contourArea, reverse = True) #sort the contours to start looking at largest

# https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
for contour in contours:
    # approximate contours
    epsilon = .1 * cv2.arcLength(contour, True)
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # pick largest contour with approximately four corners (close enough for most scans probably)
    if len(approx_contour) == 4:
        scan_contour = approx_contour
        break

contours_img = cv2.drawContours(img_to_transform, [scan_contour], -1, (255, 128, 0), 2)
if args["debug"]:
    cv2.imwrite("Contours.png", contours_img)

# apply transform
transformed_img = four_corner_transform(orig, scan_contour.reshape(4,2)*ratio)
transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)

output_name = args["name"]
if output_name is None:
    output_name = "Transformed.png"
cv2.imwrite(output_name, imutils.resize(transformed_img, height = 650))
