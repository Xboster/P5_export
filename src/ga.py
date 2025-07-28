import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        mutation_rate = 0.02  # 2% chance per tile
        left = 1
        right = width - 1

        for y in range(height):
            for x in range(left, right):
                tile = genome[y][x]
                if tile in {"m", "v", "f"}:
                    continue  # Don't mutate special tiles
                if random.random() < mutation_rate:
                    # Mutate this tile
                    genome[y][x] = random.choice(options)
        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                # Uniform crossover: 50% chance to take tile from other parent
                if random.random() < 0.5:
                    new_genome[y][x] = other.genome[y][x]

        # Mutate and return a new Individual_Grid
        mutated_genome = self.mutate(new_genome)
        return (Individual_Grid(mutated_genome),)


    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())

        coefficients = dict(
            meaningfulJumpVariance=0.8,  # encourages jumps
            negativeSpace=0.4,           # balance emptiness
            pathPercentage=0.6,          # should be playable
            emptyPercentage=0.3,         # penalize excessive emptiness
            linearity=-0.4,              # less linear = more interesting
            solvability=3.0              # solvability is crucial
        )

        penalties = 0

        # Penalty: Too many stairs
        stairs_count = len([de for de in self.genome if de[1] == "6_stairs"])
        if stairs_count > 4:
            penalties -= (stairs_count - 4) * 0.5

        # Penalty: Unplayable design (e.g., pipes without tops)
        pipe_tops = len([de for de in self.genome if de[1] == "7_pipe"])
        if pipe_tops == 0:
            penalties -= 1.0

        # Bonus: Has at least 1 platform
        platforms = len([de for de in self.genome if de[1] == "1_platform"])
        if platforms > 0:
            penalties += 0.5

        # Bonus: Variety of DEs
        unique_types = len(set(de[1] for de in self.genome))
        penalties += 0.05 * unique_types

        # Penalty: Wide holes
        wide_holes = [de for de in self.genome if de[1] == "0_hole" and de[2] > 3]
        penalties -= len(wide_holes) * 0.5

        # Penalty: Tall pipes
        tall_pipes = [de for de in self.genome if de[1] == "7_pipe" and de[2] > 3]
        penalties -= len(tall_pipes) * 0.3


        self._fitness = sum(coefficients[m] * measurements[m] for m in coefficients) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        mutation_rate = 0.2

        # Mutation: Modify an element
        if random.random() < mutation_rate and len(new_genome) > 0:
            idx = random.randint(0, len(new_genome) - 1)
            de = new_genome[idx]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()

            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                new_de = (
                    offset_by_upto(x, width // 6, 1, width - 2) if choice < 0.33 else x,
                    de_type,
                    offset_by_upto(y, 4, 1, height - 1) if 0.33 <= choice < 0.66 else y,
                    not breakable if choice >= 0.66 else breakable
                )
            elif de_type == "2_enemy":
                # Enemies only go on the ground (height - 2)
                new_de = (offset_by_upto(x, 5, 1, width - 2), de_type)
            elif de_type == "3_coin":
                y = de[2]
                new_de = (offset_by_upto(x, 5, 1, width - 2), de_type, offset_by_upto(y, 2, 1, height - 2))
            else:
                # Do nothing fancy for other types
                pass

            new_genome[idx] = new_de

        # Mutation: Add an element (small chance)
        if random.random() < 0.05:
            new_genome.append(Individual_DE.random_individual().genome[0])

        # Mutation: Remove an element (small chance)
        if random.random() < 0.05 and len(new_genome) > 1:
            del new_genome[random.randint(0, len(new_genome) - 1)]

        heapq.heapify(new_genome)
        return new_genome


    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1) if self.genome else 0
        pb = random.randint(0, len(other.genome) - 1) if other.genome else 0
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_DE

def generate_successors(population, elite_count=1, tournament_k=3, tournament_prob=0.5):
    pop_size = len(population)
    results = []

    # --- Precompute and cache fitness ---
    fitness_map = [(ind, ind.fitness()) for ind in population]

    # --- Elitism ---
    sorted_population = sorted(fitness_map, key=lambda x: x[1], reverse=True)
    elites = [copy.deepcopy(ind) for ind, _ in sorted_population[:elite_count]]

    # --- Roulette Wheel Setup ---
    fitnesses = [f for _, f in fitness_map]
    min_fitness = min(fitnesses)
    if min_fitness < 0:
        fitnesses = [f - min_fitness for f in fitnesses]
    total_fitness = sum(fitnesses)
    probabilities = [(f / total_fitness if total_fitness > 0 else 1 / pop_size) for f in fitnesses]

    cumulative_probs = []
    cumulative = 0
    for p in probabilities:
        cumulative += p
        cumulative_probs.append(cumulative)

    def roulette_select():
        r = random.random()
        for i, cp in enumerate(cumulative_probs):
            if r <= cp:
                return fitness_map[i][0]
        return fitness_map[-1][0]

    def tournament_select(k=tournament_k):
        contenders = random.sample(fitness_map, k)
        return max(contenders, key=lambda item: item[1])[0]

    # --- Generate Children ---
    while len(results) < (pop_size - elite_count):
        select = tournament_select if random.random() < tournament_prob else roulette_select
        parent1 = select()
        parent2 = select()
        while parent1 is parent2:
            parent2 = select()

        children = parent1.generate_children(parent2)
        for child in children:
            if len(results) < (pop_size - elite_count):
                results.append(child)
            else:
                break

    # --- Add elites ---
    results.extend(elites)

    return results



def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
