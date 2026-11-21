"""Small checks for aim calculations and deciding which way to dodge."""
import math
import unittest

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bot"))
from brain import Bot, ballistic


class AimTests(unittest.TestCase):
    def test_gun_leads_crossing_target_and_laser_tracks_current_position(self):
        bot = Bot.__new__(Bot)
        bot.world = SimpleNamespace(los=lambda *args: True)
        bot.p = dict(lead=1, aim_tol=1, fire_range=1)
        bot.on = dict(fire_early=False)
        me = dict(aim=[0, 0])
        gun = dict(stats=dict(speed=252, range=230))
        laser = dict(stats=dict(range=280, charge=0.75), cooldown=0)
        for speed in (-12, 12):
            target = dict(pos=[80, 0, 0], vel=[0, speed, 0])
            for secondary in (None, laser):
                kinds = ["gun", "laser" if secondary else None]
                yaw, _, _, _ = bot.shoot(me, target, kinds, [gun, secondary], 0, 0, 0, 5, 7)
                self.assertAlmostEqual(yaw, 0 if secondary else math.atan2(speed, 252))

    def test_mortar_reaches_target_at_both_angles(self):
        for height in (-8, 0, 12):
            for high in (False, True):
                with self.subTest(height=height, high=high):
                    pitch = ballistic(72, 22, 80, height, high)
                    time = 80 / (72 * math.cos(pitch))
                    self.assertAlmostEqual(72 * math.sin(pitch) * time - 11 * time**2, height)

    def test_mortar_rejects_unreachable_target(self):
        self.assertIsNone(ballistic(72, 22, 300, 0, False))


class DodgeTests(unittest.TestCase):
    def setUp(self):
        self.bot = Bot.__new__(Bot)
        self.bot.strafe = 1

    def direction(self, start, velocity):
        shot = dict(kind="gun", pos=start, vel=velocity)
        return self.bot.bullet_threat([shot], 0, 0, 0, 9)

    def test_moves_away_from_either_side(self):
        for side in (-1, 1):
            self.assertEqual(self.direction((-60, side, 4), (252, 0, 0)), (0, -side))

    def test_centered_shot_uses_strafe_side(self):
        for side in (-1, 1):
            self.bot.strafe = side
            self.assertEqual(self.direction((-60, 0, 4), (252, 0, 0)), (0, side))

    def test_ignores_receding_overhead_and_wide_shots(self):
        for start in ((60, 0, 4), (-60, 0, 14), (-60, 8, 4)):
            self.assertIsNone(self.direction(start, (252, 0, 0)))


class ScenarioTests(unittest.TestCase):
    def test_active_defender_preserves_every_command_except_fire(self):
        from unittest.mock import patch
        from combat_trials import Trial, Match, WEAPONS

        trial = Trial.__new__(Trial)
        trial.aim = trial.duel = trial.frozen = False
        defender = SimpleNamespace(id=1)
        action = dict(move=[.3, -.7], boost=True, jump=True, fire=[True, True],
                      aim=[1.2, .2], reload=True, tick=12)
        with patch.object(Match, "_apply", autospec=True) as apply:
            for trial.weapon in WEAPONS:
                trial._apply(defender, action)
                apply.assert_called_with(trial, defender, dict(action, fire=[False, False]))
                self.assertEqual(action["fire"], [True, True])

    def test_all_weapons_have_paired_dodges_and_a_duel_at_each_range(self):
        from combat_trials import cases_for_seed

        scheduled = list(cases_for_seed())
        cases = set(scheduled)
        self.assertEqual(len(scheduled), len(cases))
        for weapon in ("gun", "mortar", "laser", "lightning"):
            distances = (15, 28, 40) if weapon == "lightning" else (30, 80, 150)
            for distance in distances:
                with self.subTest(weapon=weapon, distance=distance):
                    self.assertIn((weapon, distance, "duel", 2, "both"), cases)
                    for setup in ("open", "cover"):
                        for population in (2, 16):
                            for movement in ("normal", "frozen"):
                                self.assertIn((weapon, distance, setup, population, movement), cases)


if __name__ == "__main__":
    unittest.main()
