import cv2
import tensorflow.keras as keras
import os
import numpy as np
import time


model = keras.models.load_model("sudoku_model_8.h5")
board = [[0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0]]
board_valid = [[0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0]]


label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_cell_value(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(img, (128, 128))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.array([[0, -2, 0], [-2, 10, -2], [0, -2, 0]])
    kernel_1 = np.ones((5, 5), np.uint8)
    gray = cv2.dilate(gray, kernel_1, iterations=1)
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.bitwise_not(gray)
    kernel = np.array([[0, -2, 0], [-1, 10, -1], [0, -2, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    kernel_1 = np.ones((5, 5), np.uint8)
    gray = cv2.dilate(gray, kernel_1, iterations=1)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.resize(gray, (64, 64))
    gray2 = gray.copy()
    gray = gray.reshape(1, 64 , 64 ,1)
    predict = model(gray, training=False)
    d = label[np.argmax(predict)]
    save_img(gray2, d, np.max(predict))
    return d
def save_img(img, d , k):
    _, _, files = next(os.walk("image"))
    t = len(files)
    filename = "image/eff_"+ str(t) + "_"+ str(d) +"_"+ str(k)+ ".png"
    cv2.imwrite(filename, img)

'''
    solve sudoku by using backtracking
'''
def solve(bo):
    find = find_empty(bo)
    # print_board(board)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(9):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(9):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True


def print_board(bo):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    for i in range(9):
        for j in range(9):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None



def fill_board(img):
    for x in range(9):
        for y in range(9):
            t = +1
            d = x*50
            r = y*50
            crop_img = img[d+7:d-7 + 50, r+7: r-7+50]
            d = int(get_cell_value(crop_img))
            board[x][y] = d
            board_valid[x][y] = d

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh
def get_contours(img):
    """
    returns a tuple containing the contour, its area and its corners
    """
    cnts, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    result = (None, None)

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        # skip if the area is not large enough
        if area < img.size / 3:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

        # skip if the shape is not connected
        if not cv2.isContourConvex(approx):
            continue

        if area > max_area and len(approx) == 4:
            max_area = area
            result = (cnt, approx)

    return result
def sort_corners(corners):
    """
    order is upper-left, upper-right, lower-left, lower-right
    """
    crns = [(c[0][0], c[0][1]) for c in corners]
    # tuple is sorted by its first parameter
    crns.sort()

    def sort_crn(crn):
        return crn[1]

    left_crns = crns[0:2]
    right_crns = crns[2:4]

    left_crns.sort(key=sort_crn)
    right_crns.sort(key=sort_crn)

    return [left_crns[0], right_crns[0], left_crns[1], right_crns[1]]
def get_board(img, corners):
    """
    returns the board between given corners in grayscale.
    board is 450x450 pixels
    """
    pst1 = np.float32(corners)
    pst2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pst1, pst2)
    img_warp = cv2.warpPerspective(img, matrix, (450, 450))
    return img_warp
def blend_non_transparent(background_img, overlay_img):

    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]
    overlay_mask = cv2.threshold(gray_overlay, 80, 255, cv2.THRESH_BINARY)[1]

    overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (background_img * (1 / 255.0)) * \
        (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def fill_text(bo):
    digits_img = np.zeros((450, 450, 3), np.uint8)
    for i in range(9):
        for j in range(9):
            x = i * 50
            y = j * 50
            if board_valid[j][i] == 0:
                cv2.putText(digits_img,
                            str(bo[j][i]),
                            (x + 15, y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX ,
                            1,
                            (0, 255, 0),
                            3)
    return digits_img

img = cv2.imread("board3.png")
def main(img):
    thresh = preprocess(img)
    contour, corners = get_contours(thresh)
    if contour is not None:
        corners = sort_corners(corners)
        cnt_img = img.copy()

        for crn in corners:
            cv2.circle(cnt_img, crn, 5, (189, 70, 189), -1)

        board_img = get_board(img, corners)
        fill_board(board_img)
        solve(board)
        digits_img = fill_text(board)
        height, width = img.shape[:2]
        pst1 = np.float32(
            [[0, 0], [450, 0], [0, 450], [450, 450]])
        pst2 = np.float32(corners)
        matrix = cv2.getPerspectiveTransform(pst1, pst2)
        img_warp = cv2.warpPerspective(digits_img, matrix, (width, height))
        result = blend_non_transparent(img, img_warp)
        cv2.imshow("result", result)
fps = 0
total_frames = 0
fps_wait = time.time()
cap = cv2.VideoCapture(1)

while True:
    succes, img = cap.read()

    main(img)
    cv2.putText(img,
                "{:.0f} fps".format(fps),
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 255, 0),
                1,
                cv2.LINE_AA)

    cv2.imshow("input", img)

    total_frames += 1

    cur_time = time.time()
    time_diff = cur_time - fps_wait

    if time_diff > 0.5:
        fps = total_frames / (time_diff)
        total_frames = 0
        fps_wait = cur_time

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()