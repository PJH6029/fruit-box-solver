import time
import cv2
import numpy as np
import pyautogui
import glob
import os
import pytesseract
from pynput.mouse import Button, Controller

def capture_screen():
    """Capture entire screen as a BGR image."""
    w, h = pyautogui.size()
    img = pyautogui.screenshot(region=(0, 0, w, h))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

class Board:
    DIGIT_MATCH_THRESH = 0.6

    def __init__(
        self,
        play_template_path: str = "templates/play.png",
        reset_template_path: str = "templates/reset.png",
        digit_templates_dir: str = "templates"
    ):
        # play/reset
        self.play_tpl  = cv2.imread(play_template_path)
        self.reset_tpl = cv2.imread(reset_template_path)

        # load digit templates (0–9)
        self.digit_templates = {}
        for tpl_path in glob.glob(os.path.join(digit_templates_dir, "[0-9].png")):
            digit = int(os.path.basename(tpl_path)[0])
            img   = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
            self.digit_templates[digit] = img

        self.region    = None      # (x,y,w,h) of the green border window
        self.play      = None      # callable to click Play
        self.reset     = None      # callable to click Reset
        self.board     = None      # final 2D list of ints

    def _find_game_region(self, screen: np.ndarray):
        """Find the green-bordered game window."""
        hsv    = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        lowerG = np.array([40, 60, 60])
        upperG = np.array([80, 255, 255])
        mask   = cv2.inRange(hsv, lowerG, upperG)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("Cannot find game window!")
        c = max(cnts, key=cv2.contourArea)
        self.region = cv2.boundingRect(c)

    def set_play_button_from_screen(self, screen: np.ndarray):
        """Locate Play & Reset and store click lambdas."""
        if self.region is None:
            self._find_game_region(screen)
        x0,y0,w0,h0 = self.region
        win = screen[y0:y0+h0, x0:x0+w0]

        def _match_center(tpl):
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < 0.7:
                # raise RuntimeError("Template not found!")
                return None
            tx, ty = maxloc
            th, tw = tpl.shape[:2]
            return (x0 + tx + tw//2, y0 + ty + th//2)

        play_pt  = _match_center(self.play_tpl)
        self.play  = lambda: pyautogui.click(play_pt)
    
    def set_reset_button_from_screen(self, screen: np.ndarray):
        """Locate Play & Reset and store click lambdas."""
        if self.region is None:
            self._find_game_region(screen)
        x0,y0,w0,h0 = self.region
        win = screen[y0:y0+h0, x0:x0+w0]

        def _match_center(tpl):
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < 0.7:
                # raise RuntimeError("Template not found!")
                return None
            tx, ty = maxloc
            th, tw = tpl.shape[:2]
            return (x0 + tx + tw//2, y0 + ty + th//2)

        reset_pt = _match_center(self.reset_tpl)
        self.reset = lambda: pyautogui.click(reset_pt)

    def set_board_from_screen(self, screen: np.ndarray):
        """Detect red apples and number them via template matching."""
        if self.region is None:
            self._find_game_region(screen)
        x0,y0,w0,h0 = self.region
        win = screen[y0:y0+h0, x0:x0+w0]

        # 1) isolate red regions
        hsv    = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0,  70, 50]); upper1 = np.array([10, 255,255])
        lower2 = np.array([170,70, 50]); upper2 = np.array([180,255,255])
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

        # 2) find each apple bounding box
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w < 20 or h < 20: 
                continue
            boxes.append((x,y,w,h))

        # 3) group into rows by y‐coordinate
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        rows, cur_row, prev_y = [], [], None
        for (x,y,w,h) in boxes:
            if prev_y is None or abs(y - prev_y) < h/2:
                cur_row.append((x,y,w,h))
                prev_y = y if prev_y is None else prev_y
            else:
                rows.append(cur_row)
                cur_row = [(x,y,w,h)]
                prev_y  = y
        if cur_row:
            rows.append(cur_row)

        # 4) for each cell, match every digit‐template
        board = []
        for row in rows:
            row = sorted(row, key=lambda b: b[0])
            vals = []
            for (x,y,w,h) in row:
                cell = win[y:y+h, x:x+w]
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

                best_digit, best_score = None, 0.0
                for digit, tpl in self.digit_templates.items():
                    # tpl is grayscale template, gray is your cell crop
                    th, tw = tpl.shape[:2]
                    ch, cw = gray.shape[:2]

                    # compute scale so template fits inside the cell
                    scale = min(cw / tw, ch / th, 1.0)
                    if scale < 1.0:
                        new_w = max(1, int(tw * scale))
                        new_h = max(1, int(th * scale))
                        tpl_rs = cv2.resize(tpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        tpl_rs = tpl

                    # now safe to match
                    res = cv2.matchTemplate(gray, tpl_rs, cv2.TM_CCOEFF_NORMED)
                    _, maxval, _, _ = cv2.minMaxLoc(res)
                    # print(f"Digit {digit} match score: {maxval:.2f}")
                    if maxval > best_score:
                        best_score = maxval
                        best_digit = digit
                # print(f"Cell: {x},{y} ({w},{h})")
                # print(f"Best match: {best_digit} ({best_score:.2f})")
                
                if best_score >= self.DIGIT_MATCH_THRESH:
                    vals.append(best_digit)
                else:
                    vals.append(None)  # no confident match

            board.append(vals)

        self.board = board

def start_game(board: Board):
    """Click Play, wait a bit to let the grid load."""
    
    screen = capture_screen()
    cv2.imwrite("screen_before_reset.png", screen)
    board.set_reset_button_from_screen(screen)
    board.reset()
    time.sleep(1)
    
    screen = capture_screen()
    cv2.imwrite("screen_after_reset.png", screen)
    board.set_play_button_from_screen(screen)
    board.play()
    time.sleep(1)

def main():
    time.sleep(2)                       # give you time to switch to the game
    
    board = Board()
    start_game(board)
    
    mouse = Controller()
    mouse.position = (0, 0)

    # now grab the grid
    screen2 = capture_screen()
    cv2.imwrite("screen_grid.png", screen2)
    board.set_board_from_screen(screen2)

    print("Detected board:")
    for row in board.board:
        print(row)
    # → you can now feed board.board into your solver()

if __name__ == "__main__":
    main()
