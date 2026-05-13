"""Snake + Q-Learning + Obstacles  |  ↑↓ скорость · R сброс · ESC выход"""

import pygame, sys, random
import numpy as np
from collections import defaultdict

# ── константы ─────────────────────────────────────────────────
GRID, CELL, PANEL = 20, 28, 210
W, H = GRID * CELL + PANEL, GRID * CELL
DIRS = [(-1,0),(0,1),(1,0),(0,-1)]

ALPHA, GAMMA = 0.15, 0.95
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.01, 0.997
MAX_STEPS = GRID * GRID * 3
OBS_COUNT, OBS_MIN, OBS_MAX = 8, 2, 4

C = {
    "bg":(12,12,18), "cell":(20,22,32), "grid":(26,28,42),
    "head":(72,224,128), "body":(36,148,76), "food":(230,65,65),
    "obs":(90,70,50), "obs_hi":(140,100,60),
    "panel":(16,17,26), "border":(38,40,62),
    "text":(210,212,225), "dim":(80,84,110), "accent":(100,180,255),
    "green":(72,224,128), "yellow":(230,190,60),
    "graph_ln":(72,200,128), "graph_bg":(22,24,36),
}

# ── Q-таблица ─────────────────────────────────────────────────
Q: dict = defaultdict(lambda: np.zeros(3))

def q_update(s, a, r, ns):
    Q[s][a] += ALPHA * (r + GAMMA * Q[ns].max() - Q[s][a])

def q_reset():
    Q.clear()


# ── среда ─────────────────────────────────────────────────────
class SnakeEnv:

    def __init__(self):
        self.obstacles = self._spawn_obs()   # генерируется один раз

    def reset(self):
        m = GRID // 2
        self.body  = [(m,m),(m,m-1),(m,m-2)]
        self.dir   = 1
        self.score = 0
        self.steps = 0
        self._spawn_food()
        return self._state()

    def _spawn_obs(self):
        m    = GRID // 2
        safe = {(r,c) for r in range(m-3,m+4) for c in range(m-4,m+5)}
        obs, placed, tries = set(), 0, 0
        while placed < OBS_COUNT and tries < 2000:
            tries += 1
            n = random.randint(OBS_MIN, OBS_MAX)
            if random.random() < 0.5:
                r = random.randint(1, GRID-2)
                cells = {(r, random.randint(1, GRID-1-n) + i) for i in range(n)}
            else:
                c = random.randint(1, GRID-2)
                cells = {(random.randint(1, GRID-1-n) + i, c) for i in range(n)}
            if cells & safe or cells & obs:
                continue
            obs |= cells; placed += 1
        return obs

    def _spawn_food(self):
        free = [(r,c) for r in range(GRID) for c in range(GRID)
                if (r,c) not in self.body and (r,c) not in self.obstacles]
        self.food = random.choice(free)

    def _lethal(self, r, c):
        return (r<0 or r>=GRID or c<0 or c>=GRID
                or (r,c) in self.body or (r,c) in self.obstacles)

    def _danger(self, d):
        dr,dc = DIRS[d]; r,c = self.body[0]
        return int(self._lethal(r+dr, c+dc))

    def _state(self):
        hr,hc = self.body[0]; fr,fc = self.food; d = self.dir
        return (self._danger(d), self._danger((d+1)%4), self._danger((d-1)%4),
                d, int(fr<hr), int(fr>hr), int(fc<hc), int(fc>hc))

    def step(self, action):
        self.dir = (self.dir + (1 if action==1 else -1 if action==2 else 0)) % 4
        dr,dc = DIRS[self.dir]; hr,hc = self.body[0]
        nh = (hr+dr, hc+dc); self.steps += 1

        if self._lethal(*nh):
            return self._state(), -10.0, True

        old_d = abs(hr-self.food[0]) + abs(hc-self.food[1])
        self.body.insert(0, nh)

        if nh == self.food:
            self.score += 1; self._spawn_food()
            return self._state(), 10.0, False

        self.body.pop()
        new_d = abs(nh[0]-self.food[0]) + abs(nh[1]-self.food[1])
        if self.steps >= MAX_STEPS:
            return self._state(), -5.0, True
        return self._state(), (1.0 if new_d < old_d else -1.5), False


# ── отрисовка ─────────────────────────────────────────────────
# позиции глаз по направлению
_EYE = {
    0: ((7,5),(CELL-10,5)),
    1: ((CELL-7,7),(CELL-7,CELL-10)),
    2: ((7,CELL-7),(CELL-10,CELL-7)),
    3: ((5,7),(5,CELL-10)),
}
F_LBL = F_VAL = F_SM = None

def _init_fonts():
    global F_LBL, F_VAL, F_SM
    F_LBL = pygame.font.SysFont("consolas", 12, bold=True)
    F_VAL = pygame.font.SysFont("consolas", 22, bold=True)
    F_SM  = pygame.font.SysFont("consolas", 11)

def draw(screen, env, episode, best, epsilon, fps, history):
    screen.fill(C["bg"])

    # поле
    for r in range(GRID):
        for c in range(GRID):
            pygame.draw.rect(screen, C["cell"], (c*CELL+1,r*CELL+1,CELL-2,CELL-2))
    for i in range(GRID+1):
        pygame.draw.line(screen,C["grid"],(i*CELL,0),(i*CELL,GRID*CELL))
        pygame.draw.line(screen,C["grid"],(0,i*CELL),(GRID*CELL,i*CELL))

    # препятствия
    for r,c in env.obstacles:
        pygame.draw.rect(screen,C["obs"],(c*CELL+2,r*CELL+2,CELL-4,CELL-4),border_radius=3)
        pygame.draw.line(screen,C["obs_hi"],(c*CELL+2,r*CELL+2),(c*CELL+CELL-5,r*CELL+2),2)
        pygame.draw.line(screen,C["obs_hi"],(c*CELL+2,r*CELL+2),(c*CELL+2,r*CELL+CELL-5),2)
        cx,cy = c*CELL+CELL//2, r*CELL+CELL//2
        pygame.draw.line(screen,C["obs_hi"],(cx-4,cy-4),(cx+4,cy+4),1)
        pygame.draw.line(screen,C["obs_hi"],(cx+4,cy-4),(cx-4,cy+4),1)

    # еда
    fr,fc = env.food
    pygame.draw.ellipse(screen,C["food"],(fc*CELL+5,fr*CELL+5,CELL-10,CELL-10))

    # змейка + градиент тела
    for i,(r,c) in enumerate(env.body):
        t = min(i/max(len(env.body),1),1.0)
        col = C["head"] if i==0 else tuple(int(v*(1-t*0.5)) for v in C["body"])
        pygame.draw.rect(screen,col,(c*CELL+2,r*CELL+2,CELL-4,CELL-4),border_radius=4)

    # глаза
    r,c = env.body[0]
    for ox,oy in _EYE[env.dir]:
        pos = (c*CELL+ox, r*CELL+oy)
        pygame.draw.circle(screen,(0,0,0),pos,3)
        pygame.draw.circle(screen,(255,255,255),pos,2)

    # панель
    px = GRID*CELL
    pygame.draw.rect(screen,C["panel"],(px,0,PANEL,H))
    pygame.draw.line(screen,C["border"],(px,0),(px,H),2)

    def stat(lbl, val, y, col=C["text"]):
        screen.blit(F_LBL.render(lbl,True,C["dim"]),(px+14,y))
        screen.blit(F_VAL.render(str(val),True,col),(px+14,y+16))

    stat("EPISODE",  episode,          18)
    stat("SCORE",    env.score,        78,  C["green"])
    stat("BEST",     best,             138, C["yellow"])
    stat("LENGTH",   len(env.body),    198)
    stat("EPSILON",  f"{epsilon:.3f}", 258, C["accent"])
    stat("FPS",      fps,              318)
    stat("Q-STATES", len(Q),           378, C["dim"])

    lx,ly = px+14, 418
    pygame.draw.rect(screen,C["obs"],(lx,ly,10,10),border_radius=2)
    screen.blit(F_SM.render(f"obstacles:{len(env.obstacles)}",True,C["dim"]),(lx+14,ly))

    # граф наград
    gx,gy,gw,gh = px+10, 450, PANEL-20, 90
    screen.blit(F_LBL.render("REWARD/EP",True,C["dim"]),(gx,gy-16))
    pygame.draw.rect(screen,C["graph_bg"],(gx,gy,gw,gh))
    pygame.draw.rect(screen,C["border"],(gx,gy,gw,gh),1)
    if len(history) >= 2:
        data = history[-80:]; mn,mx = min(data),max(data); rng = mx-mn or 1
        pts = [(gx+int(i/(len(data)-1)*gw),
                max(gy,min(gy+gh, gy+gh-int((v-mn)/rng*gh))))
               for i,v in enumerate(data)]
        pygame.draw.lines(screen,C["graph_ln"],False,pts,2)
        screen.blit(F_SM.render(f"{mx:.0f}",True,C["dim"]),(gx+2,gy+2))
        screen.blit(F_SM.render(f"{mn:.0f}",True,C["dim"]),(gx+2,gy+gh-14))

    for i,h in enumerate(["↑↓ speed","R reset Q","ESC quit"]):
        screen.blit(F_SM.render(h,True,C["dim"]),(px+14,H-50+i*14))

    pygame.display.flip()


# ── главный цикл ──────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Snake — Q-Learning")
    _init_fonts()
    clock = pygame.time.Clock()

    env = SnakeEnv()
    state = env.reset()
    epsilon, episode, best, fps, ep_reward = EPS_START, 1, 0, 30, 0.0
    history = []

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q): pygame.quit(); sys.exit()
                if e.key == pygame.K_UP:   fps = min(fps+10, 300)
                if e.key == pygame.K_DOWN: fps = max(fps-10, 5)
                if e.key == pygame.K_r:
                    q_reset(); epsilon=EPS_START; episode=1; best=0
                    history.clear(); env=SnakeEnv(); state=env.reset()

        action = random.randint(0,2) if random.random()<epsilon else int(Q[state].argmax())
        next_state, reward, done = env.step(action)
        ep_reward += reward
        q_update(state, action, reward, next_state)
        state = next_state

        if done:
            history.append(ep_reward); ep_reward = 0.0
            best = max(best, env.score)
            epsilon = max(EPS_END, epsilon*EPS_DECAY)
            episode += 1; state = env.reset()

        draw(screen, env, episode, best, epsilon, fps, history)
        clock.tick(fps)

if __name__ == "__main__":
    main()