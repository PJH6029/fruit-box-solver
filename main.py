import time
import pyautogui
import numpy as np
import cv2
from pynput.mouse import Button, Controller

class FruitBoxSolver:
    def solve(self, board):
        """
        Solve the Fruit Box game and return a sequence of moves.
        The board is a 2D array of Cell objects.
        """
        raise NotImplementedError("You must implement this method in your solver class.")

class FruitBoxNaiveSolver(FruitBoxSolver):
    pass

config = {
    'NROWS': 10,
    'NCOLS': 17,
    'debug': True,
    'min_matches': 5,  # minimum # of good matches before we consider a result
    'solver': FruitBoxNaiveSolver,  # set this to your solver class
}

class Cell:
    value = None
    x = None
    y = None

class Board:
    play = None
    reset = None
    board = None
    
    def set_buttons_from_screen(self, screen):
        self.play, self.reset = get_play_reset_buttons(screen)
    
    def set_board_from_screen(self, screen):
        self.board = get_board(screen)

def capture_screen():
    """Capture entire screen as a BGR image."""
    w, h = pyautogui.size()
    screen = pyautogui.screenshot(region=(0, 0, w, h))
    return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

def sift_score_digit(roi_gray, digit_gray, ratio_threshold=0.75):
    sift = cv2.SIFT_create()
    
    kp_roi, des_roi = sift.detectAndCompute(roi_gray, None)
    kp_dig, des_dig = sift.detectAndCompute(digit_gray, None)
    
    if des_roi is None or des_dig is None:
        return 0
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # KNN matching: For each descriptor in 'digit', find k=2 best matches in 'roi'
    matches = bf.knnMatch(des_dig, des_roi, k=2)

    good = []
    for m, n in matches:
        # "Lowe's ratio test" - only accept match if it's distinctly better than the next-best
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    
    # good matches = how many local features line up well
    return len(good)

def match_number(roi, templates, min_good_matches=8, ratio_threshold=0.75):
    best_match_score = 0
    best_number = -1
    
    for i, template in templates.items():
        if template is None:
            continue
        
        score = sift_score_digit(roi, template, ratio_threshold)
        
        if score > best_match_score:
            best_match_score = score
            best_number = i
    
    return best_number if best_match_score >= min_good_matches else -1

def extract_numbers(screen, img_dir="img"):
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    debug_img = screen.copy() if config['debug'] else None
    
    number_templates = {
        i: cv2.imread(f"{img_dir}/{i}.png", cv2.IMREAD_GRAYSCALE)
        for i in range(1, 10)
    }

    # save templates for debugging
    if debug_img is not None:
        cv2.imwrite("debug/debug_screen_gray.png", gray_screen)
        for i, template in number_templates.items():
            cv2.imwrite(f"debug/debug_template_{i}.png", template)
    
    
    contours, _ = cv2.findContours(gray_screen, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    numbers_found = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray_screen[y:y+h, x:x+w]
        number = match_number(
            roi, 
            number_templates,
            min_good_matches=config['min_matches'],
        )
        
        if debug_img is not None:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_img, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if number > 0:
            numbers_found.append((number, (x, y, w, h)))
            
            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if debug_img is not None:
        print(numbers_found)
        print(len(numbers_found))
        cv2.imwrite("debug/debug_extract_numbers.png", debug_img)
        
        
    if len(numbers_found) < config['NROWS'] * config['NCOLS']:
        print(f"Could not find all {config['NROWS']}x{config['NCOLS']} numbers.")
        return None
    return numbers_found

def get_board(screen, img_dir="img"):
    numbers = extract_numbers(screen, img_dir)
    print(numbers)
    
    board = [[Cell() for _ in range(config['NCOLS'])] for __ in range(config['NROWS'])]

def sift_find_center(screen_gray, template_gray, min_matches=8):
    """
    Returns the (center_x, center_y) of the template in the screen using SIFT,
    or None if insufficient matches or homography fails.
    """

    # 1) Create SIFT detector
    sift = cv2.SIFT_create()

    # 2) Find keypoints & descriptors
    kp_screen, des_screen = sift.detectAndCompute(screen_gray, None)
    kp_template, des_template = sift.detectAndCompute(template_gray, None)

    if des_screen is None or des_template is None:
        print("No descriptors found in screen or template.")
        return None

    # 3) Match descriptors with a Brute Force Matcher (crossCheck = True to filter matches)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_template, des_screen)  # query=template, train=screen
    matches = sorted(matches, key=lambda x: x.distance)  # sort by best (lowest distance) first

    # Filter out if not enough matches
    if len(matches) < min_matches:
        print(f"Not enough SIFT matches found: {len(matches)} / {min_matches}")
        return None

    # 4) Extract matched keypoints’ locations
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_screen[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # 5) Compute homography with RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        print("Homography could not be computed.")
        return None

    # 6) Use homography to transform the template’s corners
    h, w = template_gray.shape
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # 7) Calculate center from transformed corners
    #    For instance, just take the average corner:
    center_x = int(np.mean(transformed_corners[:,0,0]))
    center_y = int(np.mean(transformed_corners[:,0,1]))

    return (center_x, center_y)

def get_play_reset_buttons(screen, img_dir="img"):
    """
    Locate 'play' and 'reset' buttons in the screenshot using SIFT feature matching.
    Returns (play_center, reset_center), each either (x,y) or None.
    """
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    # If debug, copy the original BGR for drawing circles
    debug_img = screen.copy() if config['debug'] else None

    positions = {}
    for img_name in ["play", "reset"]:
        template_gray = cv2.imread(f"{img_dir}/{img_name}.png", cv2.IMREAD_GRAYSCALE)
        if template_gray is None:
            print(f"Could not read template {img_name}.png from {img_dir}/")
            positions[img_name] = None
            continue

        center = sift_find_center(
            gray_screen,
            template_gray,
            min_matches=config['min_matches']
        )
        
        if center is None:
            positions[img_name] = None
            print(f"SIFT could not locate {img_name}.")
        else:
            positions[img_name] = center
            
            # draw debug circle
            if debug_img is not None:
                print(f"SIFT found {img_name} at {center}.")
                cv2.circle(debug_img, center, 10, (0, 255, 0), 2)
                cv2.putText(debug_img, img_name, (center[0] + 10, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Save debug image so you can see if detection is correct
    if debug_img is not None:
        cv2.imwrite("debug/debug_sift_buttons.png", debug_img)
        
        print("Play button center:", positions["play"])
        print("Reset button center:", positions["reset"])
    return positions["play"], positions["reset"]

def start_game(play_pos, reset_pos):
    mouse = Controller()
    mouse.position = reset_pos
    time.sleep(0.1)
    mouse.click(Button.left)
    time.sleep(0.1)
    mouse.position = play_pos
    mouse.click(Button.left)
    time.sleep(0.5)

def execute_solution(solution):
    pass

def main():
    time.sleep(2)
    screen_bgr = capture_screen()

    cv2.imwrite("debug/screen.png", screen_bgr)

    board = Board()
    board.set_buttons_from_screen(screen_bgr)
    
    start_game(board.play, board.reset)
    
    screen_bgr = capture_screen()
    board.set_board_from_screen(screen_bgr)
    
    solver = config['solver']()
    solution = solver.solve(board.board)
    execute_solution(solution)
    
    

if __name__ == "__main__":
    main()
