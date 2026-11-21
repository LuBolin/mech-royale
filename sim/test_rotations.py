"""A watched shortcut and a protected lane around the same wall."""
import gzip
import json
import math
import unittest

import numpy as np

from bake_routes import ROOT, bake
from arena_routes import Routes
from encounters import Encounter
from rotation import departure_slack
from tracking import clear_mortar_arc
from world import World


class RotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A 27 m wall runs parallel to the direct east-west crossing.
        w = World.__new__(World)
        w.nx, w.ny, w.x0, w.y0 = 81, 81, -20, -40
        w.bounds = (-20, 60, -40, 40)
        w.ground = np.zeros((81, 81))
        w.top = w.ground.copy()
        w.top[44:47, 27:54] = 15
        w.slab_hi = w.trunk = np.full_like(w.top, -1e9)
        w.slab_lo = np.full_like(w.top, 1e9)
        w.steep = w.tmask = np.zeros_like(w.top, bool)
        w._h, w._slo, w._shi, w._tmask = (a.ravel().tolist() for a in (w.top, w.slab_lo, w.slab_hi, w.tmask))
        links, layers, cover, heights, _ = bake(w)
        empty = np.empty(0)
        cls.routes = Routes(w, links, layers, cover, heights, empty, empty, empty, 15)
        with gzip.open(ROOT/"kit/check/stream.jsonl.gz", "rt") as f:
            cls.stats = json.loads(next(f))["config"]["mech"]

    def crossing(self, watcher_y, seconds):
        eye = (20, watcher_y, 7)
        route = self.routes.plan((0, 0, 0), (40, 0), "travel", [eye], (0, 0, 100), self.stats,
                                 travel_budget=seconds)
        self.assertEqual(route[-1][:2], (40, 0))
        travel = exposure = 0.0
        for a, b in zip(route, route[1:]):
            dt = math.dist(a[:3], b[:3])/self.stats["walk_speed"]
            travel += dt
            if self.routes.world.los((a.x+b.x)/2, (a.y+b.y)/2, self.stats["height"]*0.65, *eye):
                exposure += dt
        return travel, exposure

    def test_watch_direction_and_deadline(self):
        detour = self.crossing(-30, 20)
        shortcut = self.crossing(-30, 3.7)
        swapped = self.crossing(30, 20)
        self.assertLess(shortcut[0], 3.7)
        self.assertGreater(detour[0], shortcut[0])
        self.assertLess(detour[1], shortcut[1])
        self.assertLess(swapped[0], detour[0])
        self.assertLess(swapped[1], detour[1])

    def test_outward_chase_loses_slack(self):
        me = dict(pos=(0, 0, 0), in_zone=True, stats=self.stats)
        zone = dict(center=(0, 0), radius=100, next_center=(60, 0), next_radius=20,
                    shrink_start=30, shrink_end=40)
        def slack(t, speed):
            target = dict(pos=(speed, 0, 0), vel=(speed, 0, 0))
            return departure_slack(self.routes.world, me, zone, t, target, 5)
        self.assertGreater(slack(28, 12), slack(28, -12))
        self.assertGreater(slack(28, -12), 0)
        self.assertLess(slack(36, -12), 0)

    def test_side_angle_requires_a_screened_approach(self):
        def route(target, other):
            return self.routes.plan((20, 0, 0), (0, 0), "opportunity", [target, other],
                                    (0, 0, 100), self.stats)
        self.assertFalse(route((20, -30, 4.5), (22, -30, 4.5)))
        screened = route((0, -30, 4.5), (20, 30, 4.5))
        self.assertGreater(len(screened), 1)
        end = screened[-1]
        w = self.routes.world
        self.assertTrue(w.los(end.x, end.y, end.z+self.stats["muzzle_height"], 0, -30, 4.5))
        self.assertFalse(w.los(20, 30, 7, end.x, end.y, end.z+self.stats["height"]*0.65))

    def test_beam_between_two_enemies_marks_a_duel(self):
        encounter = Encounter()
        enemies = [dict(id=1, pos=(0, -20, 0), hp=1000, aim=(0, 0)),
                   dict(id=2, pos=(40, -20, 0), hp=1000, aim=(math.pi, 0))]
        obs = dict(t=10, self=dict(id=0, pos=(20, -35, 0)), events=[], sounds=[],
                   beams=[dict(by=1, age=0, path=[[1, -20, 7], [40, -20, 4.5]])])
        self.assertEqual(encounter.observe(obs, enemies, self.routes.world), (1, 2))
        obs.update(t=13, beams=[])
        self.assertEqual(encounter.observe(obs, enemies, self.routes.world), ())

    def test_two_clear_shooters_force_a_retreat_until_healed(self):
        w = self.routes.world
        west = dict(id=1, pos=(-10, -20, 0), aim=(0, 0))
        east = dict(id=2, pos=(50, -20, 0), aim=(math.pi, 0))
        def tactic(encounter, position, hp, t=10):
            me = dict(id=0, pos=position, hp=hp, max_hp=1000, stats=self.stats)
            return encounter.choose(dict(t=t, self=me), w, None, [west, east], (), {})
        encounter = Encounter()
        self.assertEqual(tactic(encounter, (20, -20, 0), 500), "retreat")
        west["aim"], east["aim"] = (math.pi, 0), (0, 0)
        self.assertEqual(tactic(encounter, (20, -20, 0), 500, 11), "recover")
        self.assertIsNone(tactic(encounter, (20, -20, 0), 900, 12))
        # North of the wall a shooter directly south has no lane, leaving a single one.
        beside = dict(id=1, pos=(-10, 20, 0), aim=(0, 0))
        south = dict(id=2, pos=(20, -20, 0), aim=(math.pi/2, 0))
        me = dict(id=0, pos=(20, 20, 0), hp=500, max_hp=1000, stats=self.stats)
        self.assertIsNone(Encounter().choose(dict(t=10, self=me), w, None, [beside, south], (), {}))
        south["pos"] = (50, 20, 0)
        south["aim"] = (math.pi, 0)
        self.assertEqual(Encounter().choose(dict(t=10, self=me), w, None, [beside, south], (), {}), "retreat")

    def test_mortar_needs_an_arc_that_clears_the_wall(self):
        # 40 m north across the 15 m wall, at 40 m/s under 30 m/s2: two firing solutions.
        low, high = (math.asin(0.75)/2, (math.pi-math.asin(0.75))/2)
        w = self.routes.world
        self.assertFalse(clear_mortar_arc(w, (20, -20, 7), math.pi/2, low, 40, 30, 40))
        self.assertTrue(clear_mortar_arc(w, (20, -20, 7), math.pi/2, high, 40, 30, 40))
        self.assertTrue(clear_mortar_arc(w, (45, -20, 7), math.pi/2, low, 40, 30, 40))


if __name__ == "__main__":
    unittest.main()
