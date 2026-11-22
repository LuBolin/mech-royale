"""The same observation-only encoder and action vocabulary are used in training and play."""
import math

import numpy as np
from brain import Bot as Planner

TAU = math.tau
KINDS = ("gun", "mortar", "laser", "lightning")
TEACHERS = ("ours", "hermes", "giddy", "azing", "kanya", "zlyuan", "joonk", "mike",
            "osca", "noteth", "nikhil", "prakash", "arnav", "himanshu", "anthony", "whymiso",
            "overcooked", "onepizza", "fraun", "baseline")
# Move: planner, stand, 16 headings, next zone, nearest pickup, nearest cube, nearest actor.
# Aim: planner, current facing, 4 actors, 2 crates, movement heading, 16 scan directions.
SIZES = (22, 4, 25, 2, 2, 2, 2, 2, 5, 3, 4)
NAMES = ("move", "speed", "aim", "boost", "jump", "fire0", "fire1", "reload", "take", "drop", "pick")
D = 384


def unit(angle):
    return (math.cos(angle), math.sin(angle))


def direction(dx, dy):
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length else (0., 0.)


def diff(a, b):
    return (a-b+math.pi) % TAU - math.pi


def nearest(items, pos, count):
    return sorted(items, key=lambda e: (math.dist(pos, e["pos"]), e["id"]))[:count]


class Encoder:
    def __init__(self, hello):
        self.planner = Planner(hello)
        self.world = self.planner.world
        self.upgrades = {u["id"]: i for i, u in enumerate(hello["config"]["upgrades"])}
        self.pickup_radius = hello["config"]["loot"]["pickup_radius"]

    def encode(self, obs):
        me, zone = obs["self"], obs["zone"]
        x, y, z = pos = me["pos"]
        yaw, pitch = me["aim"]
        cy, sy = unit(yaw)
        base = self.planner.act(obs)
        bm = base.get("move", (0, 0))
        ba = base.get("aim", me["aim"])
        actors = nearest(obs["mechs"], pos, 4)
        crates = nearest(obs["crates"], pos, 2)
        cubes = nearest(obs["cubes"], pos, 2)
        pickups = nearest(obs["pickups"], pos, 4)
        drops = nearest(obs.get("drops", []), pos, 2)

        def local(dx, dy):
            return (cy*dx + sy*dy, -sy*dx + cy*dy)

        def rel(q):
            a, b = local(q[0]-x, q[1]-y)
            return [a/100, b/100, (q[2]-z)/30, math.hypot(a,b)/100]

        def vec(items):
            return direction(items[0]["pos"][0]-x, items[0]["pos"][1]-y) if items else (0., 0.)

        heads = [unit(yaw + k*TAU/16) for k in range(16)]
        moves = [tuple(bm), (0., 0.), *heads, direction(zone["next_center"][0]-x, zone["next_center"][1]-y),
                 vec(pickups), vec(cubes), vec(actors)]
        aims = [tuple(ba), tuple(me["aim"])]
        muzzle = me["stats"].get("muzzle_height", 7)
        for group, n in ((actors, 4), (crates, 2)):
            for j in range(n):
                if j >= len(group):
                    aims.append(None)
                    continue
                q = group[j]
                px, py, pz = q["pos"]
                height = {"mech":4.5, "npc":1.75}.get(q.get("kind"), 1.2)
                aims.append((math.atan2(py-y, px-x), math.atan2(pz+height-z-muzzle, max(.01, math.hypot(px-x,py-y)))))
        aims += [(math.atan2(bm[1], bm[0]), 0.)] + [(yaw+k*TAU/16, 0.) for k in range(16)]
        choices = (obs.get("offer") or {}).get("choices", [])
        take_ids = [q["id"] if math.hypot(q["pos"][0]-x,q["pos"][1]-y) <= self.pickup_radius
                    and abs(q["pos"][2]-z) <= 4 else None for q in pickups]
        vx, vy, vz = me["vel"]
        f = [obs["t"]/480, obs["alive"]/16, x/600, y/600, z/50, cy, sy, pitch,
             me["hp"]/max(1,me["max_hp"]), me["energy"]/max(1,me["max_energy"]),
             me["grounded"], me["in_zone"], me["cubes"]/12, *[v/20 for v in local(vx,vy)], vz/20]
        for name in ("center", "next_center"):
            a, b = local(zone[name][0]-x, zone[name][1]-y)
            f.extend((a/300, b/300))
        f.extend((zone["radius"]/600, zone["next_radius"]/600, (zone["shrink_start"]-obs["t"])/120,
                  (zone["shrink_end"]-obs["t"])/120, zone["dps"]/100))
        for w in me["weapons"]:
            w = w or {}
            f.extend([w.get("kind")==k for k in KINDS] + [w.get("charge",0), w.get("ammo",0)/16,
                     w.get("cooldown",0), w.get("reload",0)])
        f.extend([uid in me["upgrades"] for uid in self.upgrades][:22])
        for j in range(3):
            idx = self.upgrades.get(choices[j]["id"],-1) if j < len(choices) else -1
            f.extend(idx == k for k in range(22))
        for j in range(4):
            if j >= len(actors):
                f.extend([0]*15)
                continue
            a = actors[j]
            f.extend([1, *rel(a["pos"]), *[v/20 for v in local(*a["vel"][:2])], a["vel"][2]/20,
                      a.get("hp",0)/max(1,a.get("max_hp",1)), a.get("kind")=="mech", a.get("charging",False),
                      *[k in a.get("weapons",()) for k in KINDS]])
        for group, count in ((pickups,4), (cubes,2), (crates,2), (drops,2)):
            for j in range(count):
                if j < len(group):
                    q = group[j]
                    f.extend([1,*rel(q["pos"]),*[q.get("kind")==k for k in KINDS],q.get("lands_in",0)/30])
                else:
                    f.extend([0]*10)
        projs = nearest(obs["projectiles"], pos, 3)
        for j in range(3):
            if j < len(projs):
                q = projs[j]
                f.extend([1,*rel(q["pos"]),*[v/200 for v in local(*q["vel"][:2])],q["vel"][2]/200])
            else:
                f.extend([0]*8)
        for e in (min(obs["sounds"], key=lambda q:q["age"], default={}),
                  next((q for q in reversed(obs["events"]) if q["type"]=="damaged"), {})):
            bearing = e.get("bearing")
            f.extend([bearing is not None, *(unit(bearing-yaw) if bearing is not None else (0,0)),
                      e.get("dist")=="near", e.get("amount",0)/300])
        beams = obs.get("beams", [])
        f.extend(rel(beams[-1]["path"][0]) if beams and beams[-1]["path"] else [0]*4)
        for dx,dy in heads[::2]:
            f.append(self.world.clear(x,y,z,dx,dy,12,me["stats"].get("step_up",1.2)))
        fire = base.get("fire", (False,False))
        f.extend([*local(*bm), *unit(ba[0]-yaw), ba[1], base.get("boost",False), base.get("jump",False),
                  *fire, base.get("reload",False), base.get("take") is not None, (base.get("pick",-1)+1)/3])
        data = np.zeros(D,np.float32)
        data[:len(f)] = f
        np.clip(data,-8,8,out=data)
        return data, (moves, aims, take_ids, choices, base, actors, pos, yaw)


def labels(ctx, action, goal=None, target=None):
    moves, aims, ids, choices, base, actors, pos, yaw = ctx
    move = action.get("move",(0,0))
    speed = math.hypot(*move)
    # Equal candidate directions prefer the planner's continuous command.
    d = direction(*move)
    m = min(range(len(moves)),key=lambda i:math.dist(direction(*moves[i]),d))
    speed_id = min(range(4),key=lambda i:abs([math.hypot(*moves[m]),0,.45,1][i]-speed))
    aim = action.get("aim",aims[1])
    ai = min((i for i,a in enumerate(aims) if a is not None),
             key=lambda i:diff(aims[i][0],aim[0])**2+(aims[i][1]-aim[1])**2)
    fire = action.get("fire",(False,False))
    take = action.get("take")
    obj = ids.index(take)+1 if take is not None and take in ids else 0
    aux_goal = -100 if goal is None else round(diff(math.atan2(goal[1]-pos[1],goal[0]-pos[0]),yaw)*16/TAU) % 16
    aux_target = next((i for i,a in enumerate(actors) if a["id"]==target),-100)
    return [m,speed_id,ai,int(action.get("boost",False)),int(action.get("jump",False)),int(fire[0]),int(fire[1]),
            int(action.get("reload",False)),obj if any(i is not None for i in ids) else -100,
            action.get("drop",-1)+1,(action.get("pick",-1)+1) if choices else -100,aux_goal,aux_target]


def decode(ctx, heads, tick):
    moves, aims, ids, choices, base, *_ = ctx
    # Mask unavailable object slots before selecting the deterministic argmax.
    if isinstance(heads, np.ndarray):
        heads = list(np.split(heads, np.cumsum(SIZES)[:-1]))
        heads[2] = np.where([v is not None for v in aims],heads[2],-np.inf)
        heads[8] = np.where([True]+[i is not None for i in ids]+[False]*(4-len(ids)),heads[8],-np.inf)
        heads[10] = np.where(np.arange(4)<=len(choices),heads[10],-np.inf)
        heads = [int(np.argmax(v)) for v in heads]
    m,s,ai,boost,jump,f0,f1,reload,take,drop,pick = heads
    v = moves[m]
    if s:
        mag = (0,0,.45,1)[s]
        v = [k*mag for k in direction(*v)]
    aim = aims[ai] or aims[0]
    act = dict(tick=tick, move=list(v), aim=[diff(aim[0],0),float(aim[1])], boost=bool(boost),
               jump=bool(jump), fire=[bool(f0),bool(f1)], reload=bool(reload))
    if take and take<=len(ids) and ids[take-1] is not None:
        act["take"] = ids[take-1]
    if drop:
        act["drop"] = drop-1
    if 0<pick<=len(choices):
        act["pick"] = pick-1
    return act
