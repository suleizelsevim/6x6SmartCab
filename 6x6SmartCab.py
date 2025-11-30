import gymnasium as gym
from gym import spaces
import numpy as np
import random
from io import StringIO
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
from PIL import Image


# ============================================================
#                CUSTOM SMARTCAB 6x6 ENVIRONMENT
# ============================================================

class TaxiEnv6x6(gym.Env):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self):
        super().__init__()

        self.rows = 6
        self.cols = 6

        # ------------------ FORBIDDEN CELLS ------------------
        # Bu hücrelere TAKSI de YOLCU da giremez / konamaz
        self.forbidden_cells = {
            (1, 1),
            (2, 3),
            (4, 4),
        }

        # ------------------------ DUVARLAR ------------------------
        # Hücre kenarlarındaki duvarlar
        # N: yukarı, S: aşağı, E: sağ, W: sol
        self.walls = {
    (0, 4): {"S": True},   # E -> sağa duvar
    (0, 5): {"S": True},   # S -> aşağı duvar
    (1, 4): {"S": True},   # W -> sola duvar
    (1, 5): {"S": True},   # N -> yukarı duvar
    (4, 0): {"E": True},   
    (5, 0): {"E": True},
    (4, 1): {"E": True}, 
    (5, 1): {"E": True},
}


        # Eksik yönleri False ile tamamla ve duvarı komşuya da yansıt
        self._normalize_and_propagate_walls()

        # --------------------- ACTION / OBS SPACE ---------------------
        # 0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff
        self.action_space = spaces.Discrete(6)

        self.n_pos = self.rows * self.cols          # 36
        self.n_pass = self.rows * self.cols + 1     # 37 (0..35 hücre, 36 = takside)
        self.n_dest = self.rows * self.cols         # 36

        self.observation_space = spaces.Discrete(self.n_pos * self.n_pass * self.n_dest)

        # İç durum
        self.taxi_row = 0
        self.taxi_col = 0
        self.passenger_state = 0   # 0..35 -> hücre indexi, 36 -> takside
        self.dest_index = 0        # 0..35

        self.reset()

    # ---------------- DUVARLARI DÜZELT / PROPAGATE ----------------

    def _normalize_and_propagate_walls(self):
        # Tüm hücreler için sözlük kaydı oluştur
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    self.walls[(r, c)] = {}
                for d in ["N", "S", "E", "W"]:
                    if d not in self.walls[(r, c)]:
                        self.walls[(r, c)][d] = False

        # Karşılıklı hücrelere duvarı yansıt
        for r in range(self.rows):
            for c in range(self.cols):
                if self.walls[(r, c)]["N"] and r > 0:
                    self.walls[(r - 1, c)]["S"] = True
                if self.walls[(r, c)]["S"] and r < self.rows - 1:
                    self.walls[(r + 1, c)]["N"] = True
                if self.walls[(r, c)]["E"] and c < self.cols - 1:
                    self.walls[(r, c + 1)]["W"] = True
                if self.walls[(r, c)]["W"] and c > 0:
                    self.walls[(r, c - 1)]["E"] = True

    # ------------------------- YARDIMCI -------------------------

    def idx_to_rc(self, idx):
        return divmod(idx, self.cols)

    def rc_to_idx(self, r, c):
        return r * self.cols + c

    def encode(self, tr, tc, p, d):
        i = tr
        i = i * self.cols + tc
        i = i * self.n_pass + p
        i = i * self.n_dest + d
        return i

    def decode(self, state):
        d = state % self.n_dest
        state //= self.n_dest
        p = state % self.n_pass
        state //= self.n_pass
        tc = state % self.cols
        tr = state // self.cols
        return tr, tc, p, d

    def _random_free_cell_index(self):
        while True:
            idx = random.randint(0, self.rows * self.cols - 1)
            r, c = self.idx_to_rc(idx)
            if (r, c) not in self.forbidden_cells:
                return idx

    # --------------------------- RESET --------------------------

    def reset(self):
        self.state_visit = {}
        self.last_position = None
        # Taksi
        taxi_idx = self._random_free_cell_index()
        self.taxi_row, self.taxi_col = self.idx_to_rc(taxi_idx)

        # Yolcu
        passenger_idx = self._random_free_cell_index()
        self.passenger_state = passenger_idx

        # Hedef
        while True:
            dest_idx = self._random_free_cell_index()
            if dest_idx != passenger_idx:
                self.dest_index = dest_idx
                break

        return self.encode(self.taxi_row, self.taxi_col,
                           self.passenger_state, self.dest_index)

    # ---------------------------- STEP --------------------------

    def step(self, action):
        assert self.action_space.contains(action)

        # Loop tracking for the current state
        key = (self.taxi_row, self.taxi_col, self.passenger_state)
        self.state_visit[key] = self.state_visit.get(key, 0) + 1

        old_r, old_c = self.taxi_row, self.taxi_col
        r, c = old_r, old_c

        # Reward constants
        step_penalty = -0.5       # Her normal adımda verilen küçük ceza (agent boş boş gezinmesin diye)
        wall_penalty = -10.0      # Duvara çarpınca verilen büyük ceza (duvarlardan uzak durmayı öğrenmesi için)
        illegal_penalty = -20.0   # Geçersiz pickup/dropoff işlemlerinde verilen ceza (yanlış yerde işlem yapmasını engeller)
        pickup_reward = 5.0       # Yolcuyu başarılı şekilde almak için ödül (görevin doğru başladığını pekiştirir)
        success_reward = 50.0     # Yolcuyu doğru yerde indirme ödülü (episode’un amacını temsil eden büyük ödül)
        shaping = 0.5             # Hedefe/yolcuya yaklaşmayı ödüllendiren shaping değeri (öğrenmeyi hızlandırır)


        reward = step_penalty
        done = False

        in_taxi = (self.passenger_state == self.rows * self.cols)

        # ------------------ MOVEMENT ------------------
        if action == 0:  # SOUTH
            if not self.walls[(r, c)]["S"]:
                nr = r + 1
                if nr < self.rows and (nr, c) not in self.forbidden_cells:
                    r = nr
                else:
                    reward += wall_penalty
            else:
                reward += wall_penalty

        elif action == 1:  # NORTH
            if not self.walls[(r, c)]["N"]:
                nr = r - 1
                if nr >= 0 and (nr, c) not in self.forbidden_cells:
                    r = nr
                else:
                    reward += wall_penalty
            else:
                reward += wall_penalty

        elif action == 2:  # EAST
            if not self.walls[(r, c)]["E"]:
                nc = c + 1
                if nc < self.cols and (r, nc) not in self.forbidden_cells:
                    c = nc
                else:
                    reward += wall_penalty
            else:
                reward += wall_penalty

        elif action == 3:  # WEST
            if not self.walls[(r, c)]["W"]:
                nc = c - 1
                if nc >= 0 and (r, nc) not in self.forbidden_cells:
                    c = nc
                else:
                    reward += wall_penalty
            else:
                reward += wall_penalty

        # Apply new position
        self.taxi_row, self.taxi_col = r, c

        # ------------------ PICKUP & DROPOFF ------------------
        if action == 4:  # Pickup
            if (not in_taxi) and self.passenger_state == self.rc_to_idx(r, c):
                self.passenger_state = self.rows * self.cols
                reward += pickup_reward
            else:
                reward += illegal_penalty

        elif action == 5:  # Dropoff
            if in_taxi and self.dest_index == self.rc_to_idx(r, c):
                reward += success_reward
                done = True
                self.passenger_state = self.dest_index
            else:
                reward += illegal_penalty

        # ------------------ REWARD SHAPING ------------------
        target_idx = self.passenger_state if not in_taxi else self.dest_index
        tr, tc = self.idx_to_rc(target_idx)

        old_dist = abs(old_r - tr) + abs(old_c - tc)
        new_dist = abs(r - tr) + abs(c - tc)

        if new_dist < old_dist:
            reward += shaping
        elif new_dist > old_dist:
            reward -= shaping
        else:
            reward -= 5  # prevents "standing still" indirectly

        # ------------------ LOOP PENALTY ------------------
        if self.state_visit[key] > 3:
            reward -= 2  # soft anti-loop
            
            
        # ----- ANTI BACKTRACKING: tekrar eski hücreye dönüyorsa ceza -----
        if self.last_position is not None:
            if (r, c) == self.last_position:
                reward -= 3.0   # ping-pong kırıcı ceza


        # Return next state
        next_state = self.encode(self.taxi_row, self.taxi_col,
                                 self.passenger_state, self.dest_index)
        visits = self.state_visit.get(next_state, 0)
        reward -= visits * 0.5
        self.state_visit[next_state] = visits + 1

        self.last_position = (old_r, old_c)
        return next_state, reward, done, {}


# ============================================================
#                  Q-LEARNING TRAINING (OPTIMIZED)
# ============================================================

def train_agent(
    episodes=150000,
    alpha=0.05,          # daha küçük öğrenme oranı
    gamma=0.995,         # geleceğe daha çok önem
    eps_start=1.0,
    eps_min=0.1,
    eps_decay_episodes=150000,  # tüm episode boyunca yavaş azalsın
    qfile="q_smartcab_6x6_opt.npy"
):
    env = TaxiEnv6x6()
    q = np.zeros((env.observation_space.n, env.action_space.n))

    epsilon = eps_start

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 500   # daha uzun epizod

        while not done and steps < max_steps:
            # epsilon-greedy policy
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(q[s]))

            ns, r, done, _ = env.step(a)

            old = q[s, a]
            q[s, a] = (1 - alpha) * old + alpha * (r + gamma * np.max(q[ns]))

            s = ns
            total_reward += r
            steps += 1

        # epsilon decay
        if ep < eps_decay_episodes:
            epsilon = eps_start - (eps_start - eps_min) * (ep / eps_decay_episodes)
        else:
            epsilon = eps_min

        if ep % 5000 == 0:
            print(f"Episode {ep}/{episodes} - epsilon={epsilon:.3f} - last_total_reward={total_reward:.2f}")

    # Q-table kaydet
    np.save(qfile, q)
    print("\nQ-table kaydedildi:", qfile)

    # Küçük bir özet yazdır
    print("\n=== Q-Table Summary ===")
    print("Shape:", q.shape)
    print("First 5 states' Q-values:")
    for i in range(5):
        print(f"State {i}: {q[i]}")

    return q


# ============================================================
#              EĞİTİLMİŞ AGENT'İ DEĞERLENDİR
# ============================================================

def evaluate_agent(q, n_episodes=20):
    env = TaxiEnv6x6()
    success_count = 0
    steps_list = []
    rewards_list = []

    for ep in range(n_episodes):
        s = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 500

        while not done and steps < max_steps:
            a = int(np.argmax(q[s]))
            s, r, done, _ = env.step(a)
            total_reward += r
            steps += 1

        if done:
            success_count += 1

        steps_list.append(steps)
        rewards_list.append(total_reward)

    print("\n=== Evaluation Results ===")
    print(f"Episodes       : {n_episodes}")
    print(f"Success count  : {success_count}")
    print(f"Success rate   : {success_count / n_episodes:.2f}")
    print(f"Avg steps      : {sum(steps_list) / len(steps_list):.2f}")
    print(f"Avg total reward: {sum(rewards_list) / len(rewards_list):.2f}")


# ============================================================
#               EĞİTİLMİŞ AGENT'İ GÖZLEMLE (RENDER)
# ============================================================

def run_trained_agent(q, n_episodes=3, delay=0.2):
    env = TaxiEnv6x6()

    for ep in range(1, n_episodes + 1):
        s = env.reset()
        done = False
        steps = 0
        max_steps = 500

        print(f"\n===== TEST EPISODE {ep} =====\n")

        while not done and steps < max_steps:
            env.render()
            a = int(np.argmax(q[s]))
            s, r, done, _ = env.step(a)
            steps += 1
            time.sleep(delay)


def save_gif_from_agent(q, filename="smartcab.gif",
                        episodes=3, delay=0.12, max_steps=40):
    """
    Matplotlib animasyonunu direkt GIF olarak kaydeder.
    DPI/reshape hataları tamamen düzeltilmiştir.
    """
    env = TaxiEnv6x6()

    frames = []  # Pillow Image listesi

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # 600x600 px → reshape hatası olmaz

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        steps = 0
        total_reward = 0
        last_reward = 0

        while not done and steps < max_steps:

            ax.clear()

            # === GRID + FORBIDDEN ===
            for r in range(env.rows):
                for c in range(env.cols):
                    face = "red" if (r, c) in env.forbidden_cells else "white"
                    rect = patches.Rectangle(
                        (c, env.rows - 1 - r), 1, 1,
                        edgecolor="black", facecolor=face, linewidth=1
                    )
                    ax.add_patch(rect)

            # === WALLS ===
            for r in range(env.rows):
                for c in range(env.cols):
                    x = c
                    y = env.rows - 1 - r
                    if env.walls[(r, c)]["N"]:
                        ax.plot([x, x+1], [y+1, y+1], color="black", linewidth=3)
                    if env.walls[(r, c)]["S"]:
                        ax.plot([x, x+1], [y, y], color="black", linewidth=3)
                    if env.walls[(r, c)]["W"]:
                        ax.plot([x, x], [y, y+1], color="black", linewidth=3)
                    if env.walls[(r, c)]["E"]:
                        ax.plot([x+1, x+1], [y, y+1], color="black", linewidth=3)

            # === Destination ===
            dr, dc = env.idx_to_rc(env.dest_index)
            ax.add_patch(patches.Rectangle(
                (dc, env.rows - 1 - dr), 1, 1,
                facecolor="lightgreen", linewidth=2
            ))

            # === Passenger ===
            if env.passenger_state != env.rows * env.cols:
                pr, pc = env.idx_to_rc(env.passenger_state)
                ax.add_patch(patches.Rectangle(
                    (pc, env.rows - 1 - pr), 1, 1,
                    facecolor="lightskyblue", linewidth=2
                ))

            # === Taxi ===
            ax.add_patch(patches.Rectangle(
                (env.taxi_col, env.rows - 1 - env.taxi_row), 1, 1,
                facecolor="gold", linewidth=3
            ))

            # === Title overlay ===
            ax.set_title(
                f"Episode {ep}/{episodes} | Step {steps}\n"
                f"Reward={last_reward:.2f} | Total={total_reward:.2f}",
                fontsize=10
            )

            ax.set_xlim(0, env.cols)
            ax.set_ylim(0, env.rows)
            ax.set_aspect("equal")
            ax.axis("off")

            # --- CANVAS'TAN GÖRÜNTÜ AL ---
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame = Image.fromarray(rgba)
            frames.append(frame.convert("P", palette=Image.ADAPTIVE))

            # === Step ===
            a = int(np.argmax(q[s]))
            ns, reward, done, _ = env.step(a)
            last_reward = reward
            total_reward += reward
            s = ns
            steps += 1

    # --- GIF olarak kaydet ---
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=int(delay * 1000),
        loop=0
    )

    print(f"\nGIF kaydedildi: {filename}")


def animate_25_even_episodes(q, delay=0.12, max_steps=40):
    """
    Eğitimin genel davranışını temsil etmesi için 25 episode oynatır.
    Animasyonda step, reward ve total reward gösterilir.
    """
    TOTAL_EPISODES = 25
    env = TaxiEnv6x6()

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    for ep in range(1, TOTAL_EPISODES + 1):
        s = env.reset()
        done = False
        steps = 0
        total_reward = 0
        last_reward = 0

        print(f"\n=== ANIMATION EPISODE {ep}/{TOTAL_EPISODES} ===\n")

        while not done and steps < max_steps:
            ax.clear()

            # === GRID + FORBIDDEN ===
            for r in range(env.rows):
                for c in range(env.cols):
                    face = "red" if (r, c) in env.forbidden_cells else "white"
                    rect = patches.Rectangle(
                        (c, env.rows - 1 - r), 1, 1,
                        edgecolor="black", facecolor=face, linewidth=1
                    )
                    ax.add_patch(rect)

            # === DUVARLAR ===
            for r in range(env.rows):
                for c in range(env.cols):
                    x = c
                    y = env.rows - 1 - r

                    if env.walls[(r, c)]["N"]:
                        ax.plot([x, x+1], [y+1, y+1], color="black", linewidth=3)
                    if env.walls[(r, c)]["S"]:
                        ax.plot([x, x+1], [y, y], color="black", linewidth=3)
                    if env.walls[(r, c)]["W"]:
                        ax.plot([x, x], [y, y+1], color="black", linewidth=3)
                    if env.walls[(r, c)]["E"]:
                        ax.plot([x+1, x+1], [y, y+1], color="black", linewidth=3)

            # === HEDEF === (yeşil)
            dr, dc = env.idx_to_rc(env.dest_index)
            ax.add_patch(patches.Rectangle(
                (dc, env.rows - 1 - dr), 1, 1,
                facecolor="lightgreen", linewidth=2
            ))

            # === YOLCU === (mavi)
            if env.passenger_state != env.rows * env.cols:
                pr, pc = env.idx_to_rc(env.passenger_state)
                ax.add_patch(patches.Rectangle(
                    (pc, env.rows - 1 - pr), 1, 1,
                    facecolor="lightskyblue", linewidth=2
                ))

            # === TAKSİ === (sarı)
            ax.add_patch(patches.Rectangle(
                (env.taxi_col, env.rows - 1 - env.taxi_row), 1, 1,
                facecolor="gold", linewidth=3
            ))

            # === BAŞLIK: EPISODE | STEP | REWARD ===
            ax.set_title(
                f"Episode {ep}/25  |  Step {steps}\n"
                f"Reward = {last_reward:.2f}   |   Total = {total_reward:.2f}",
                fontsize=11
            )

            ax.set_xlim(0, env.cols)
            ax.set_ylim(0, env.rows)
            ax.set_aspect("equal")
            ax.axis("off")

            plt.pause(delay)

            # === POLICY (greedy) ===
            a = int(np.argmax(q[s]))
            ns, reward, done, _ = env.step(a)

            last_reward = reward
            total_reward += reward

            s = ns
            steps += 1

        plt.pause(0.25)

    plt.ioff()
    plt.show()



# ============================================================
#                            MAIN
# ============================================================

if __name__ == "__main__":
    # 1) Q-learning eğitimi
    q = train_agent()

    # 2) Eğitim sonrası başarı analizi
    evaluate_agent(q, n_episodes=20)

    # 3) Terminalde ASCII görmek istersen:
    # run_trained_agent(q, n_episodes=3, delay=0.2)

    # 4) Matplotlib ile animasyon (GIF'e benzer)
    animate_25_even_episodes(q, delay=0.1, max_steps=40)
    
    save_gif_from_agent(q, filename="smartcab_6x6.gif", episodes=20, max_steps=40)




