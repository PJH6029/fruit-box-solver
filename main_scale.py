import time
import cv2
import numpy as np
import pyautogui
import glob
import os
import pytesseract
from pynput.mouse import Button, Controller

class Board:
    # how many scales to try between min and max
    MSCALE_STEPS = 15
    # play/reset match threshold
    BUTTON_THRESH = 0.7
    # digit match threshold
    DIGIT_THRESH = 0.6

    def __init__(
        self,
        play_template_path: str = "templates/play.png",
        reset_template_path: str = "templates/reset.png",
        digit_templates_dir: str = "templates"
    ):
        # load play/reset in color
        self.play_tpl  = cv2.imread(play_template_path)
        self.reset_tpl = cv2.imread(reset_template_path)

        # load 0–9 templates as gray
        self.digit_templates = {}
        for p in glob.glob(os.path.join(digit_templates_dir, "[0-9].png")):
            digit = int(os.path.basename(p)[0])
            self.digit_templates[digit] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        self.region = None
        self.play   = None
        self.reset  = None
        self.board  = None

    def _find_game_region(self, screen):
        hsv    = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        lowerG = np.array([40, 60, 60])
        upperG = np.array([80, 255, 255])
        mask   = cv2.inRange(hsv, lowerG, upperG)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("No green window found")
        c = max(cnts, key=cv2.contourArea)
        self.region = cv2.boundingRect(c)

    def _match_multiscale(self, image, tpl, thresh):
        """
        Try matching `tpl` at many scales on `image`. Returns (loc, scale).
        """
        h0, w0 = tpl.shape[:2]
        best = (-1, None, None)  # (score, loc, scale)
        for scale in np.linspace(0.5, 2.0, self.MSCALE_STEPS):
            tw, th = int(w0*scale), int(h0*scale)
            if tw < 8 or th < 8 or image.shape[0] < th or image.shape[1] < tw:
                continue
            tpl_rs = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
            res    = cv2.matchTemplate(image, tpl_rs, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (mx, loc, scale)
        score, loc, scale = best
        if score < thresh:
            raise RuntimeError(f"Template match failed (best={score:.2f})")
        return loc, scale

    def set_play_button_from_screen(self, screen):
        if self.region is None:
            self._find_game_region(screen)
        x0,y0,w0,h0 = self.region
        win = screen[y0:y0+h0, x0:x0+w0]

        # Play
        loc, scale = self._match_multiscale(win, self.play_tpl, self.BUTTON_THRESH)
        th, tw = int(self.play_tpl.shape[0]*scale), int(self.play_tpl.shape[1]*scale)
        px, py = x0 + loc[0] + tw//2, y0 + loc[1] + th//2
        self.play = lambda: pyautogui.click((px, py))

    def set_reset_button_from_screen(self, screen):
        if self.region is None:
            self._find_game_region(screen)
        x0,y0,w0,h0 = self.region
        win = screen[y0:y0+h0, x0:x0+w0]
        
        # Reset
        loc, scale = self._match_multiscale(win, self.reset_tpl, self.BUTTON_THRESH)
        th, tw = int(self.reset_tpl.shape[0]*scale), int(self.reset_tpl.shape[1]*scale)
        rx, ry = x0 + loc[0] + tw//2, y0 + loc[1] + th//2
        self.reset = lambda: pyautogui.click((rx, ry))

    def set_board_from_screen(self, screen):
        if self.region is None:
            self._find_game_region(screen)
        x0,y0,w0,h0 = self.region
        win = screen[y0:y0+h0, x0:x0+w0]

        # isolate red apples
        hsv      = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
        lower1   = np.array([0,70,50]);   upper1 = np.array([10,255,255])
        lower2   = np.array([170,70,50]); upper2 = np.array([180,255,255])
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(m1|m2, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get bounding boxes
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c)>200]
        boxes.sort(key=lambda b:(b[1],b[0]))

        # group into rows
        rows, cur, prev_y = [], [], None
        for (x,y,w,h) in boxes:
            if prev_y is None or abs(y-prev_y)<h/2:
                cur.append((x,y,w,h))
                prev_y = y if prev_y is None else prev_y
            else:
                rows.append(cur); cur=[(x,y,w,h)]; prev_y=y
        if cur: rows.append(cur)

        board = []
        for row in rows:
            row.sort(key=lambda b:b[0])
            vals = []
            for (x,y,w,h) in row:
                cell = cv2.cvtColor(win[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

                best_digit, best_score = None, -1
                for digit, tpl in self.digit_templates.items():
                    # scale template to cell height
                    th, tw = tpl.shape
                    scale = h / th
                    # keep it inside cell width
                    if tw*scale > w:
                        scale = w / tw
                    tw2, th2 = int(tw*scale), int(th*scale)
                    if tw2<8 or th2<8:
                        continue
                    tpl_rs = cv2.resize(tpl, (tw2, th2), interpolation=cv2.INTER_AREA)
                    res    = cv2.matchTemplate(cell, tpl_rs, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, _ = cv2.minMaxLoc(res)
                    if mx > best_score:
                        best_score, best_digit = mx, digit

                vals.append(best_digit if best_score>=self.DIGIT_THRESH else None)
            board.append(vals)

        self.board = board
        
def capture_screen():
    """Capture entire screen as a BGR image."""
    w, h = pyautogui.size()
    img = pyautogui.screenshot(region=(0, 0, w, h))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


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
