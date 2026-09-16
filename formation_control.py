import numpy as np
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
from tqdm import tqdm
from collections import defaultdict
import subprocess


############################################## CORE FUNCTION ####################################################
def calc_formn_input(A,P,k,formn,i,neighbour,kp=0.1):
    sum_ = 0
    pi = P[k][i]
    ri = formn[i]
    for j in neighbour:
        aij = A[i,j]
        pj = P[k][j]
        rj = formn[j]

        sum_ += aij * ( (pi - pj) - (ri - rj) )

    sum_ += kp * (pi - ri)

    return sum_


############################################## HELPER FUNCTION ####################################################

def plot_graph(G, state=4, out_dir=None):
    pos = nx.spring_layout(G, seed=state)

    plt.figure(figsize=(6,4))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="mediumpurple",
        node_size=500,
        edge_color="black",
        font_size=12,
        font_weight="bold"
    )
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, "Agent_Graph.png"), dpi=300)
    
    plt.close()

def make_movie(P, formn, A, out_dir, dpi=200):

    os.makedirs(out_dir, exist_ok=True)

    N = len(P[0])

    all_positions = np.vstack(P)

    xmin = min(formn[:, 0].min(), all_positions[:, 0].min())
    xmax = max(formn[:, 0].max(), all_positions[:, 0].max())

    ymin = min(formn[:, 1].min(), all_positions[:, 1].min())
    ymax = max(formn[:, 1].max(), all_positions[:, 1].max())

    xpad = 0.1 * (xmax - xmin)
    ypad = 0.1 * (ymax - ymin)

    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad


    fig, ax = plt.subplots(figsize=(7, 5))
    fig2, ax2 = plt.subplots(figsize=(7,5))

    edges = []

    for i in range(N):
        for j in range(i + 1, N):

            if A[i, j] != 0:
                edges.append((i, j))

    #Animation frames
    for k, positions in tqdm(enumerate(P)):

        ax.clear()
        #2. Agent trajectories
        for i in range(N):

            trajectory = np.array(P[:k + 1])[:, i, :]

            # ax.plot(
            #     trajectory[:, 0],
            #     trajectory[:, 1],
            #     linewidth=0.7,
            #     alpha=0.35
            # )

       
        #3.Communication links
        for i, j in edges:

            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                linestyle=':',
                linewidth=0.8,
                alpha=0.25,
            )

        #4.Current agent positions
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            marker='x',
            color='mediumpurple',
            s=75,
            linewidths=2.8,
            label='Agents'
        )


        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(
            'X',
            fontsize=18,
            fontweight='bold'
        )

        ax.set_ylabel(
            'Y',
            fontsize=18,
            fontweight='bold'
        )

        ax.set_title(
            f'Timestep = {k}',
            fontsize=20,
            fontweight='bold'
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(
            axis='both',
            labelsize=13
        )

        ax.legend(
            frameon=False,
            fontsize=11
        )

        fig.tight_layout()

        #5.Save frame
        filename = os.path.join(
            out_dir,
            f'timestep_{k:04d}.png'
        )

        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches='tight'
        )

    plt.close(fig)


    print(f"Saved {len(P)} frames to {out_dir}")


#Getting formation coordinates for all 26 english alphabets

LETTERS = {

    "A": [
        [(0.05, 0.0), (0.50, 1.0), (0.95, 0.0)],
        [(0.25, 0.40), (0.75, 0.40)]
    ],

    "B": [
        [(0.10, 0.0), (0.10, 1.0)],
        [(0.10, 1.0), (0.65, 1.0), (0.85, 0.85),
         (0.85, 0.60), (0.65, 0.50), (0.10, 0.50)],
        [(0.10, 0.50), (0.65, 0.50), (0.85, 0.40),
         (0.85, 0.15), (0.65, 0.0), (0.10, 0.0)]
    ],

    "C": [
        [(0.90, 0.85), (0.75, 1.0), (0.35, 1.0),
         (0.10, 0.75), (0.10, 0.25), (0.35, 0.0),
         (0.75, 0.0), (0.90, 0.15)]
    ],

    "D": [
        [(0.10, 0.0), (0.10, 1.0)],
        [(0.10, 1.0), (0.55, 1.0), (0.85, 0.80),
         (0.85, 0.20), (0.55, 0.0), (0.10, 0.0)]
    ],

    "E": [
        [(0.90, 1.0), (0.10, 1.0), (0.10, 0.0), (0.90, 0.0)],
        [(0.10, 0.50), (0.70, 0.50)]
    ],

    "F": [
        [(0.10, 0.0), (0.10, 1.0), (0.90, 1.0)],
        [(0.10, 0.50), (0.70, 0.50)]
    ],

    "G": [
        [(0.90, 0.80), (0.75, 1.0), (0.35, 1.0),
         (0.10, 0.75), (0.10, 0.25), (0.35, 0.0),
         (0.75, 0.0), (0.90, 0.20), (0.90, 0.45),
         (0.55, 0.45)]
    ],

    "H": [
        [(0.10, 0.0), (0.10, 1.0)],
        [(0.90, 0.0), (0.90, 1.0)],
        [(0.10, 0.50), (0.90, 0.50)]
    ],

    "I": [
        [(0.10, 1.0), (0.90, 1.0)],
        [(0.50, 1.0), (0.50, 0.0)],
        [(0.10, 0.0), (0.90, 0.0)]
    ],

    "J": [
        [(0.10, 1.0), (0.90, 1.0)],
        [(0.70, 1.0), (0.70, 0.20),
         (0.55, 0.0), (0.30, 0.0), (0.10, 0.20)]
    ],

    "K": [
        [(0.10, 0.0), (0.10, 1.0)],
        [(0.90, 1.0), (0.10, 0.50), (0.90, 0.0)]
    ],

    "L": [
        [(0.10, 1.0), (0.10, 0.0), (0.90, 0.0)]
    ],

    "M": [
        [(0.10, 0.0), (0.10, 1.0), (0.50, 0.50),
         (0.90, 1.0), (0.90, 0.0)]
    ],

    "N": [
        [(0.10, 0.0), (0.10, 1.0), (0.90, 0.0),
         (0.90, 1.0)]
    ],

    "O": [
        [(0.35, 0.0), (0.10, 0.25), (0.10, 0.75),
         (0.35, 1.0), (0.65, 1.0), (0.90, 0.75),
         (0.90, 0.25), (0.65, 0.0), (0.35, 0.0)]
    ],

    "P": [
        [(0.10, 0.0), (0.10, 1.0)],
        [(0.10, 1.0), (0.60, 1.0), (0.85, 0.80),
         (0.85, 0.60), (0.60, 0.50), (0.10, 0.50)]
    ],

    "Q": [
        [(0.35, 0.0), (0.10, 0.25), (0.10, 0.75),
         (0.35, 1.0), (0.65, 1.0), (0.90, 0.75),
         (0.90, 0.25), (0.65, 0.0), (0.35, 0.0)],
        [(0.60, 0.25), (0.95, -0.05)]
    ],

    "R": [
        [(0.10, 0.0), (0.10, 1.0)],
        [(0.10, 1.0), (0.60, 1.0), (0.85, 0.80),
         (0.85, 0.60), (0.60, 0.50), (0.10, 0.50)],
        [(0.55, 0.50), (0.90, 0.0)]
    ],

    "S": [
        [(0.85, 0.85), (0.65, 1.0), (0.30, 1.0),
         (0.10, 0.80), (0.10, 0.60), (0.30, 0.50),
         (0.70, 0.50), (0.90, 0.40), (0.90, 0.20),
         (0.70, 0.0), (0.30, 0.0), (0.10, 0.15)]
    ],

    "T": [
        [(0.10, 1.0), (0.90, 1.0)],
        [(0.50, 1.0), (0.50, 0.0)]
    ],

    "U": [
        [(0.10, 1.0), (0.10, 0.25),
         (0.25, 0.0), (0.75, 0.0),
         (0.90, 0.25), (0.90, 1.0)]
    ],

    "V": [
        [(0.10, 1.0), (0.50, 0.0), (0.90, 1.0)]
    ],

    "W": [
        [(0.05, 1.0), (0.25, 0.0), (0.50, 0.65),
         (0.75, 0.0), (0.95, 1.0)]
    ],

    "X": [
        [(0.10, 1.0), (0.90, 0.0)],
        [(0.90, 1.0), (0.10, 0.0)]
    ],

    "Y": [
        [(0.10, 1.0), (0.50, 0.50), (0.90, 1.0)],
        [(0.50, 0.50), (0.50, 0.0)]
    ],

    "Z": [
        [(0.10, 1.0), (0.90, 1.0),
         (0.10, 0.0), (0.90, 0.0)]
    ],
}


#2.CALCULATE LENGTH OF A STROKE
def stroke_length(stroke):

    points = np.asarray(stroke, dtype=float)

    return np.sum(
        np.linalg.norm(
            np.diff(points, axis=0),
            axis=1
        )
    )


#3. SAMPLE N POINTS ALONG ONE STROKE
def sample_stroke(stroke, n):

    points = np.asarray(stroke, dtype=float)

    if n <= 1:
        return points[[0]]

    segment_lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    cumulative = np.concatenate([
        [0],
        np.cumsum(segment_lengths)
    ])

    total_length = cumulative[-1]

    if total_length == 0:
        return np.repeat(points[[0]], n, axis=0)

    distances = np.linspace(
        0,
        total_length,
        n
    )

    x = np.interp(
        distances,
        cumulative,
        points[:, 0]
    )

    y = np.interp(
        distances,
        cumulative,
        points[:, 1]
    )

    return np.column_stack([x, y])


#4. DISTRIBUTE N AGENTS BETWEEN MULTIPLE STROKES
def sample_letter(strokes, N):

    if N < len(strokes):
        raise ValueError(
            f"N={N} is too small. "
            f"This letter requires at least {len(strokes)} agents."
        )

    lengths = np.array([
        stroke_length(stroke)
        for stroke in strokes
    ])

    #ideal number of agents per stroke
    ideal = N * lengths / lengths.sum()

    #start with floor
    counts = np.floor(ideal).astype(int)

    #ensure every non-zero stroke gets at least one agent
    counts[counts == 0] = 1

    #correct total
    while counts.sum() < N:

        remainder = ideal - counts

        idx = np.argmax(remainder)

        counts[idx] += 1

    while counts.sum() > N:

        #remove from the stroke with the largest allocation
        idx = np.argmax(counts)

        if counts[idx] > 1:
            counts[idx] -= 1
        else:
            break

    #sample each stroke
    samples = []

    for stroke, n in zip(strokes, counts):

        pts = sample_stroke(
            stroke,
            n
        )

        samples.append(pts)

    return np.vstack(samples)



#5. NORMALIZE TO COMMON RECTANGULAR BOX
def normalize_formation(
    points,
    width=10.0,
    height=10.0,
    center=True
):

    points = np.asarray(points, dtype=float).copy()

    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    # Avoid division by zero
    xrange_ = xmax - xmin
    yrange_ = ymax - ymin

    if xrange_ == 0:
        xrange_ = 1.0

    if yrange_ == 0:
        yrange_ = 1.0

    # Scale into [0, width] x [0, height]
    points[:, 0] = (
        (points[:, 0] - xmin)
        / xrange_
        * width
    )

    points[:, 1] = (
        (points[:, 1] - ymin)
        / yrange_
        * height
    )

    if center:
        points[:, 0] -= width / 2
        points[:, 1] -= height / 2

    return points



#6. GENERATE ONE LETTER
def make_letter(letter, N, width=10.0, height=10.0):

    letter = letter.upper()

    if letter not in LETTERS:
        raise ValueError(
            f"Unknown letter: {letter}"
        )

    points = sample_letter(
        LETTERS[letter],
        N
    )

    points = normalize_formation(
        points,
        width=width,
        height=height
    )

    return points



#7. GENERATE ALL 26 LETTERS
def make_all_letters(N, width=10.0, height=10.0):

    formations = {}

    for letter in LETTERS:

        formations[letter] = make_letter(
            letter,
            N,
            width=width,
            height=height
        )

    return formations



############################################## MAIN FUNCTION ####################################################

tol = 1
dt = 0.1
name = sys.argv[1]
name=name.upper()
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
SEED = 42

#Erdos-Renyi graph parameters
N = 20
p = 0.1

#seeding for reproducibility
np.random.seed(SEED)


#generating Erdos-Renyi connected graph
print(f"Starting with SEED = {SEED}")
G = nx.erdos_renyi_graph(n=N, p=p, seed=SEED)

while not nx.is_connected(G):
    SEED+=1
    G = nx.erdos_renyi_graph(n=N, p=p, seed=SEED)

print(f"Got connecteed graph at SEED = {SEED}")

plot_graph(G,state=4,out_dir=out_dir)

#generate initial 2d positions for the agents
init_pos = np.random.randn(N,2)

#get the formations for letters depending upon the number of agents N
formations = make_all_letters(
    N,
    width=10,
    height=10
)

#get the adjacency matrix
A = nx.adjacency_matrix(G).toarray()

#initialize a dictionary to store u_i_k
U = defaultdict(list)
#initialize a list to store the positions over timesteps
P = [init_pos]

err=[]
k = 0

#get the letter formations
forms = [formations[l] for l in name]
#iterate over formations
for formn in forms:
    maxsteps=5000
    formation_err = np.linalg.norm(P[k]-formn)
    while formation_err > tol and k < maxsteps:
        pk = []
        print(f"Step {k} || Formation Error = {formation_err : .4f} > {tol}")
        for i in range(N):
            neighbour = nx.neighbors(G, i)
            formn_input = calc_formn_input(A,P,k,formn,i,neighbour)
            next_pos = P[k][i] - dt * formn_input
            pk.append(next_pos)
        P.append(np.vstack(pk))
    
        k = k+1
        formation_err = np.linalg.norm(P[k]-formn)
    
        err.append(formation_err)

    for _ in range(20):
        P.append(np.vstack(pk))
        k+=1
        err.append(formation_err)
    
print("Formation Executed Successfully !")

#plot formation error
plt.plot(err)
plt.ylabel("Formation Error")
plt.xlabel("Timestep (t)")
sns.despine()
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "FormationError.png"), dpi=300)
plt.close()

print("Generating Frames for Animated Movie...")
#make frames for generating animation movie
make_movie(P, formn, A, os.path.join(out_dir, "frames"))


print("Stitching Frames...")
subprocess.run([
    "ffmpeg",
    "-y",
    "-framerate", "30",
    "-i", f"{out_dir}/frames/timestep_%04d.png",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "12",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    f"{out_dir}/simulation.mp4"
], check=True)

print(f"Done, Results saved at : {out_dir}")