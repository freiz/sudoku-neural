# Based on: http://norvig.com/sudoku.html
from tensorflow import keras
import numpy as np
import time


def cross(A, B):
    """Cross product of elements in A and elements in B."""
    return [a + b for a in A for b in B]


digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s], [])) - set([s]))
             for s in squares)
search_count = 0


def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False  ## (Fail if we can't assign d to square s.)
    return values


def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values
    values[s] = values[s].replace(d, '')
    if len(values[s]) == 0:
        return False
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False
        elif len(dplaces) == 1:
            if not assign(values, dplaces[0], d):
                return False
    return values


def display(values):
    """Display these values as a 2-D grid."""
    width = 1 + max(len(values[s]) for s in squares)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join([''.join(values[r + c].center(width) + ('|' if c in '36' else ''))
                       for c in cols]))
        if r in 'CF':
            print(line)
    print()


def solve(grid, prediction=None):
    prob = None
    if prediction is not None:
        prob = {}
        for i, p in enumerate(squares):
            prob[p] = prediction[i]

    return search(parse_grid(grid), prob)


def search(values, prob=None):
    "Using depth-first search and propagation, try all possible values."
    global search_count
    search_count += 1
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in squares):
        return values
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)

    if prob is not None:
        search_order_dict = {}
        for d in values[s]:
            search_order_dict[d] = prob[s][int(d) - 1]
        search_order = [k for k, v in sorted(search_order_dict.items(), key=lambda item: item[1], reverse=True)]
    else:
        search_order = values[s]

    return some(search(assign(values.copy(), s, d), prob) for d in search_order)


def some(seq):
    """Return some element of seq that is true."""
    for e in seq:
        if e:
            return e
    return False


def predict(model, problems):
    n = len(problems)
    tensor = np.array([int(c) for c in ''.join(problems).replace('.', '0')]).reshape((n, 9, 9, 1))
    prediction = model.predict(tensor)

    return prediction


def solve_all(problems, model=None):
    if model is not None:
        prediction = predict(model, problems)
        for i, problem in enumerate(problems):
            solve(problem, prediction[i])
    else:
        for problem in problems:
            solve(problem)


def solve_all_in_file(file_path, model=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        solve_all([line.strip() for line in lines], model)


if __name__ == '__main__':
    model = keras.models.load_model('model')

    grid1 = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
    hardest = '.....6....59.....82....8....45........3........6..3.54...325..6..................'

    predict(model, [grid1])  # warm up gpu and the framework

    # solve the hardest problem
    start = time.process_time()
    search_count = 0
    prediction = predict(model, [hardest])[0]
    solution = solve(hardest, prediction)
    print(f'Model On: Solved in: {time.process_time() - start} s')
    print(f'Search Count: {search_count}')
    display(solution)

    start = time.process_time()
    search_count = 0
    solution = solve(hardest)
    print(f'Model Off: Solved in: {time.process_time() - start} s')
    print(f'Search Count: {search_count}')
    display(solution)

    # solve a list of hard problems
    with open('data/hard95.txt', 'r') as f:
        lines = f.readlines()
        problems = [line.strip() for line in lines]

    start = time.process_time()
    search_count = 0
    solve_all(problems, model)
    print(f'Model On: Solved in: {time.process_time() - start} s')
    print(f'Search Count: {search_count}')

    start = time.process_time()
    search_count = 0
    solve_all(problems)
    print(f'Model Off: Solved in: {time.process_time() - start} s')
    print(f'Search Count: {search_count}')
