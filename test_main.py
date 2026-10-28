import unittest
from xml.etree import ElementTree

from main import (DEFAULT_GRID, MAPS, evaluate, learning_svg, parse_grid, policy_svg,
                  shortest_path, summarize, train, transition)


class GridTests(unittest.TestCase):
    def test_rectangular_map(self):
        grid = parse_grid('S.#\n..G', 'tiny')
        self.assertEqual((grid.height, grid.width), (2, 3))
        self.assertEqual(grid.start, (0, 0))
        self.assertEqual(grid.goal, (1, 2))
        self.assertEqual(grid.walls, {(0, 2)})

    def test_reject_invalid_maps(self):
        for source in ['', 'S..\n.G', 'SXG', 'SSG', 'S..', 'G..']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_grid(source)

    def test_transition_rewards_and_terminal_state(self):
        grid = parse_grid('S#G\n...')
        self.assertEqual(transition(grid.start, 0, grid), (grid.start, -5.0, False))
        self.assertEqual(transition(grid.start, 1, grid), (grid.start, -5.0, False))
        self.assertEqual(transition(grid.start, 2, grid), ((1, 0), -1.0, False))
        self.assertEqual(transition((1, 2), 0, grid), (grid.goal, 100.0, True))
        self.assertEqual(transition(grid.goal, 3, grid), (grid.goal, 0.0, True))

    def test_bfs_matches_known_shortest_paths(self):
        grid = parse_grid('S.#\n..G')
        path = shortest_path(grid)
        self.assertEqual(len(path) - 1, 3)
        self.assertEqual((path[0], path[-1]), (grid.start, grid.goal))
        self.assertEqual(len(shortest_path(DEFAULT_GRID)) - 1, 8)
        detour = parse_grid(MAPS['detour'])
        self.assertGreater(len(shortest_path(detour)) - 1,
                           abs(detour.start[0] - detour.goal[0]) + abs(detour.start[1] - detour.goal[1]))
        self.assertIsNone(shortest_path(parse_grid('S#G')))


class LearningTests(unittest.TestCase):
    def test_training_is_reproducible_without_changing_global_rng(self):
        import random
        state = random.getstate()
        first = train(80, 7)
        self.assertEqual(first, train(80, 7))
        self.assertEqual(state, random.getstate())
        self.assertEqual(len(first[1]), 80)

    def test_agent_learns_a_real_shortest_path(self):
        for seed in (0, 42):
            with self.subTest(seed=seed):
                q, history = train(1000, seed)
                path, success = evaluate(q)
                self.assertTrue(success)
                self.assertEqual(path[0], DEFAULT_GRID.start)
                self.assertEqual(path[-1], DEFAULT_GRID.goal)
                self.assertEqual(len(path), len(shortest_path()))
                self.assertTrue(any(any(values) for values in q.values()))
                for state, nxt in zip(path, path[1:]):
                    self.assertEqual(abs(state[0] - nxt[0]) + abs(state[1] - nxt[1]), 1)
                    self.assertNotIn(nxt, DEFAULT_GRID.walls)
                self.assertTrue(summarize(DEFAULT_GRID, seed, history, path, success)['optimal'])

    def test_untrained_policy_reports_a_loop_and_no_successful_path_length(self):
        q = {state: [0.0] * 4 for state in DEFAULT_GRID.states()}
        path, success = evaluate(q)
        self.assertFalse(success)
        self.assertLessEqual(len(path), len(q) + 1)
        summary = summarize(DEFAULT_GRID, 1, [{'success': False}], path, success)
        self.assertIsNone(summary['learned_path_steps'])
        self.assertIsNone(summary['excess_steps'])
        self.assertFalse(summary['optimal'])

    def test_invalid_training_limits(self):
        with self.assertRaises(ValueError):
            train(0)
        with self.assertRaises(ValueError):
            train(10, max_steps=0)

    def test_generated_svgs_are_well_formed_and_escape_custom_names(self):
        grid = parse_grid('SG', '<tiny & map>')
        q, history = train(30, 0, grid)
        path, success = evaluate(q, grid)
        for source in [policy_svg(q, grid, path, success), learning_svg(history, grid.name)]:
            root = ElementTree.fromstring(source)
            self.assertEqual(root.tag, '{http://www.w3.org/2000/svg}svg')
            self.assertIn('&lt;tiny &amp; map&gt;', source)


if __name__ == '__main__':
    unittest.main()
