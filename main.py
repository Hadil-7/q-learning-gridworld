"""Learn deterministic grid navigation with tabular Q-learning; compare it with BFS."""
import argparse
from collections import deque
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import random

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SYMBOLS = ['↑', '→', '↓', '←']
MAPS = {
    'classic': '..#.G\n..##.\n.....\n.#.#.\nS....',
    'detour': 'S..#..G\n##.#.##\n...#...\n.#####.\n.......\n.###.#.\n.......',
}


@dataclass(frozen=True)
class Grid:
    rows: tuple
    start: tuple
    goal: tuple
    walls: frozenset
    name: str = 'custom'

    @property
    def height(self):
        return len(self.rows)

    @property
    def width(self):
        return len(self.rows[0])

    def states(self):
        return [(r, c) for r in range(self.height) for c in range(self.width)
                if (r, c) not in self.walls]


def parse_grid(source, name='custom'):
    rows = tuple(line.strip() for line in source.strip().splitlines() if line.strip())
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('A map must be a nonempty rectangle.')
    if any(ch not in '.#SG' for row in rows for ch in row):
        raise ValueError('Map cells must be ., #, S or G.')
    starts = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == 'S']
    goals = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == 'G']
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError('A map needs exactly one start S and one goal G.')
    walls = frozenset((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == '#')
    return Grid(rows, starts[0], goals[0], walls, name)


DEFAULT_GRID = parse_grid(MAPS['classic'], 'classic')
# Retained for readers of the original five-by-five example.
SIZE, START, GOAL, WALLS = DEFAULT_GRID.height, DEFAULT_GRID.start, DEFAULT_GRID.goal, DEFAULT_GRID.walls


def transition(state, action, grid=DEFAULT_GRID):
    if action not in range(len(ACTIONS)):
        raise ValueError('Action must be an integer from 0 to 3.')
    if state == grid.goal:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    nxt = state[0] + dr, state[1] + dc
    if not (0 <= nxt[0] < grid.height and 0 <= nxt[1] < grid.width) or nxt in grid.walls:
        return state, -5.0, False
    return nxt, (100.0 if nxt == grid.goal else -1.0), nxt == grid.goal


def shortest_path(grid=DEFAULT_GRID):
    """Breadth-first search is an independent shortest-path reference, not training data."""
    queue, parents = deque([grid.start]), {grid.start: None}
    while queue:
        state = queue.popleft()
        if state == grid.goal:
            path = []
            while state is not None:
                path.append(state)
                state = parents[state]
            return path[::-1]
        for action in range(4):
            nxt, _, _ = transition(state, action, grid)
            if nxt not in parents:
                parents[nxt] = state
                queue.append(nxt)
    return None


def greedy_action(values):
    # A fixed tie-break makes evaluation and seeded training reproducible.
    return max(range(4), key=lambda action: values[action])


def train(episodes=3000, seed=42, grid=DEFAULT_GRID, max_steps=150):
    if episodes < 1 or max_steps < 1:
        raise ValueError('Episodes and max_steps must be positive.')
    rng = random.Random(seed)
    q = {state: [0.0] * 4 for state in grid.states()}
    history = []
    for episode in range(episodes):
        state, total, done = grid.start, 0.0, False
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.8))
        for step in range(max_steps):
            action = rng.randrange(4) if rng.random() < epsilon else greedy_action(q[state])
            nxt, reward, done = transition(state, action, grid)
            target = reward if done else reward + 0.95 * max(q[nxt])
            q[state][action] += 0.2 * (target - q[state][action])
            total += reward
            state = nxt
            if done:
                break
        history.append({'episode': episode + 1, 'reward': total, 'steps': step + 1, 'success': done})
    return q, history


def evaluate(q, grid=DEFAULT_GRID, max_steps=None):
    """Follow only learned Q-values; stop at the goal, a repeated state or the step cap."""
    path, seen, state = [grid.start], {grid.start}, grid.start
    for _ in range(max_steps if max_steps is not None else len(grid.states())):
        if state == grid.goal:
            return path, True
        nxt, _, done = transition(state, greedy_action(q[state]), grid)
        path.append(nxt)
        if done:
            return path, True
        if nxt in seen:
            return path, False
        seen.add(nxt)
        state = nxt
    return path, False


def summarize(grid, seed, history, path, success):
    reference = shortest_path(grid)
    optimal_steps = len(reference) - 1 if reference else None
    steps = len(path) - 1 if success else None
    recent = history[-100:]
    return {
        'map': grid.name, 'seed': seed, 'episodes': len(history),
        'greedy_success': success, 'learned_path_steps': steps,
        'bfs_shortest_steps': optimal_steps,
        'optimal': success and steps == optimal_steps,
        'excess_steps': steps - optimal_steps if success and reference else None,
        'training_success_rate_last_100': round(sum(item['success'] for item in recent) / len(recent), 4),
    }


def policy_text(q, grid=DEFAULT_GRID):
    return '\n'.join(' '.join('#' if (r, c) in grid.walls else 'G' if (r, c) == grid.goal
                             else SYMBOLS[greedy_action(q[(r, c)])] for c in range(grid.width))
                     for r in range(grid.height))


def policy_svg(q, grid, path, success):
    cell, margin = 58, 28
    width, height = max(360, grid.width * cell + margin * 2), grid.height * cell + 110
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{margin}" y="25" font-family="sans-serif" font-size="16">{escape(grid.name)}: learned greedy policy</text>']
    for r in range(grid.height):
        for c in range(grid.width):
            state = (r, c)
            color = '#334155' if state in grid.walls else '#dcfce7' if state == grid.goal else '#dbeafe' if state == grid.start else '#f8fafc'
            x, y = margin + c * cell, 44 + r * cell
            out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#cbd5e1"/>')
            if state not in grid.walls:
                symbol = 'G' if state == grid.goal else SYMBOLS[greedy_action(q[state])] if any(q[state]) else '·'
                out.append(f'<text x="{x + cell / 2}" y="{y + 36}" text-anchor="middle" font-family="sans-serif" font-size="23">{symbol}</text>')
    points = ' '.join(f'{margin + c * cell + cell / 2},{44 + r * cell + cell / 2}' for r, c in path)
    out.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="4" stroke-opacity="0.5"/>')
    out.append(f'<text x="{margin}" y="{height - 32}" font-family="sans-serif" font-size="12">Blue line: actual rollout. Blue cell: start.</text>')
    out.append(f'<text x="{margin}" y="{height - 14}" font-family="sans-serif" font-size="12">Goal reached: {str(success).lower()}. Dots: all Q-values are zero.</text>')
    out.append('</svg>')
    return '\n'.join(out)


def learning_svg(history, title, window=50):
    values, recent, total = [], deque(), 0.0
    for item in history:
        recent.append(item['reward'])
        total += item['reward']
        if len(recent) > window:
            total -= recent.popleft()
        values.append(total / len(recent))
    low, high = min(values), max(values)
    span = high - low or 1.0
    # Keep exported curves compact while retaining both endpoints.
    stride = max(1, len(values) // 600)
    indices = sorted(set(range(0, len(values), stride)) | {len(values) - 1})
    points = ' '.join(f'{65 + i * 610 / max(1, len(values) - 1):.1f},{315 - (values[i] - low) * 250 / span:.1f}'
                      for i in indices)
    return '\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="740" height="390" viewBox="0 0 740 390">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="65" y="25" font-family="sans-serif" font-size="16">{escape(title)}</text>',
        f'<text x="65" y="46" font-family="sans-serif" font-size="12">Training return: trailing mean of up to {window} episodes</text>',
        '<path d="M65 65V315H675" fill="none" stroke="#64748b"/>',
        f'<text x="8" y="76" font-family="sans-serif" font-size="12">{high:.1f}</text>',
        f'<text x="8" y="315" font-family="sans-serif" font-size="12">{low:.1f}</text>',
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>',
        '<text x="65" y="340" font-family="sans-serif" font-size="12">1</text>',
        f'<text x="675" y="340" text-anchor="end" font-family="sans-serif" font-size="12">{len(history)}</text>',
        '<text x="370" y="365" text-anchor="middle" font-family="sans-serif" font-size="13">Episode</text>',
        '</svg>',
    ])


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', help='Comma-separated seeds for a repeated experiment; overrides --seed.')
    parser.add_argument('--map', choices=MAPS, default='classic')
    parser.add_argument('--map-file', type=Path, help='Custom ASCII rectangle using S, G, # and .')
    parser.add_argument('--all-maps', action='store_true', help='Run both built-in maps.')
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--output', type=Path, default=Path('results.json'))
    parser.add_argument('--summary', type=Path, help='Also write compact, history-free experiment metrics.')
    parser.add_argument('--svg-dir', type=Path, help='Write policy and learning-curve SVGs per run.')
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error('--episodes and --max-steps must be positive')
    if args.map_file and args.all_maps:
        parser.error('--map-file cannot be combined with --all-maps')
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(',')] if args.seeds else [args.seed]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Use distinct seeds for repeated experiments.')
        grids = [parse_grid(args.map_file.read_text(encoding='utf-8'), args.map_file.stem)] if args.map_file else [
            parse_grid(MAPS[name], name) for name in (MAPS if args.all_maps else [args.map])]
        for grid in grids:
            if shortest_path(grid) is None:
                raise ValueError(f'The goal in {grid.name} is unreachable from its start.')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    runs, metrics = [], []
    for grid in grids:
        for seed in seeds:
            q, history = train(args.episodes, seed, grid, args.max_steps)
            path, success = evaluate(q, grid)
            metric = summarize(grid, seed, history, path, success)
            metrics.append(metric)
            runs.append({**metric, 'path': path, 'history': history,
                         'q_values': {f'{r},{c}': values for (r, c), values in q.items()}})
            print(f"{grid.name} | seed {seed} | goal {success} | learned moves {metric['learned_path_steps']} | BFS {metric['bfs_shortest_steps']}")
            if len(grids) * len(seeds) == 1:
                print(policy_text(q, grid))
            if args.svg_dir:
                args.svg_dir.mkdir(parents=True, exist_ok=True)
                label = f'{grid.name}-seed-{seed}'
                (args.svg_dir / f'{label}-policy.svg').write_text(policy_svg(q, grid, path, success), encoding='utf-8')
                (args.svg_dir / f'{label}-learning.svg').write_text(learning_svg(history, label), encoding='utf-8')
    summary = {'algorithm': 'tabular Q-learning', 'episodes_per_run': args.episodes,
               'max_steps_per_episode': args.max_steps, 'seeds': seeds,
               'maps': {grid.name: list(grid.rows) for grid in grids},
               'run_count': len(metrics), 'greedy_successes': sum(item['greedy_success'] for item in metrics),
               'optimal_runs': sum(item['optimal'] for item in metrics), 'runs': metrics}
    write_json(args.output, {'summary': summary, 'runs': runs})
    if args.summary:
        write_json(args.summary, summary)
    print(f"Successful greedy rollouts: {summary['greedy_successes']}/{len(metrics)}; shortest paths: {summary['optimal_runs']}/{len(metrics)}")
    print(f'Detailed results: {args.output}')


if __name__ == '__main__':
    main()
