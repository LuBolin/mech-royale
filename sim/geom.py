"""Exact arena geometry for the simulator: a numba port of the official web client's world code
(play.joinbuilders.org, class with surfaceHeight / penetration / moveAndSlide / raycast /
canopyTransmittance). Same data, same algorithms: polygon prisms, bilinear terrain, trunk
cylinders, crown ellipsoids, 16 m broad-phase cells.

Verified against the client itself (sim/test_geom.py): 0 mismatches in 1000 seeded queries each of
surface height, sight, canopy transmittance, shot raycast and move-and-slide.

Choices where the client and protocol.md differ:
- step height comes from hello.config (`mech.step_up` 1.2; the client's world rules say 1.5);
- trunks stop sight and movement (the client's penetration() always includes them) but not shots
  (protocol: "Trees do not stop shots"; the client's raycast has a trunks flag for this);
- slopes steeper than max_slope_deg cannot be walked up (protocol §7; the client only checks spawns).
"""
import json
import math
import os

import numpy as np
from numba import njit

XF = 1e-9
BF = 16.0
NOHIT = 2.0
TRUNK_COLLIDE = True


class Geom:
    def __init__(self, map_dir, step_up=1.2, max_slope_deg=50.0):
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        t, b, r = w["terrain"], w["bounds"], w["rules"]
        raw = np.fromfile(os.path.join(map_dir, "v2", t["file"]), "<i2")
        H = raw.astype(np.float64) * t["scale"]  # client keeps float32; float64 here is a superset
        nx, ny = t["nx"], t["ny"]
        bnx, bny = -(-nx // 8), -(-ny // 8)
        hb = H.reshape(ny, nx)
        bm = np.full((bny, bnx), -np.inf)
        for j in range(bny):
            for i in range(bnx):
                bm[j, i] = hb[j * 8:(j + 1) * 8, i * 8:(i + 1) * 8].max()
        dil = bm.copy()
        dil[:, :-1] = np.maximum(dil[:, :-1], bm[:, 1:])
        dil[:-1, :] = np.maximum(dil[:-1, :], bm[1:, :])
        dil[:-1, :-1] = np.maximum(dil[:-1, :-1], bm[1:, 1:])
        self.tg = np.array([t["minX"], t["minY"], t["step"], nx, ny, bnx, bny,
                            b["minX"], b["minY"], b["maxX"], b["maxY"]], np.float64)
        pr = w["prisms"]
        edges, pes, pbox = [], [], []
        for p in pr:
            s = len(edges)
            xs, ys = [], []
            for ring in [p["ring"]] + list(p.get("holes") or []):
                n = len(ring) // 2
                for k in range(n):
                    l = (k + 1) % n
                    edges.append((ring[2 * k], ring[2 * k + 1], ring[2 * l], ring[2 * l + 1]))
                    xs.append(ring[2 * k])
                    ys.append(ring[2 * k + 1])
            pes.append((s, len(edges)))
            pbox.append((min(xs), min(ys), max(xs), max(ys)))
        fi = {n: k for k, n in enumerate(w["trees"]["fields"])}
        d = np.asarray(w["trees"]["data"], float).reshape(-1, len(fi))
        z, hgt, cb = d[:, fi["z"]], d[:, fi["height"]], d[:, fi["crownBase"]]
        v, yy = z + cb, z + hgt
        tr = np.stack([d[:, fi["x"]], d[:, fi["y"]], z, d[:, fi["trunkR"]], z + np.maximum(cb + 0.5 * (hgt - cb), 3.0),
                       d[:, fi["crownX"]] if "crownX" in fi else d[:, fi["x"]],
                       d[:, fi["crownY"]] if "crownY" in fi else d[:, fi["y"]],
                       (v + yy) / 2, d[:, fi["crownR"]], np.maximum((yy - v) / 2, 0.5)], 1)
        gx0, gy0 = b["minX"] - BF, b["minY"] - BF
        gnx = int(math.ceil((b["maxX"] - b["minX"]) / BF)) + 3
        gny = int(math.ceil((b["maxY"] - b["minY"]) / BF)) + 3
        gp = [[] for _ in range(gnx * gny)]
        gt = [[] for _ in range(gnx * gny)]

        def cells(x0, y0, x1, y1):
            a, o = max(0, int(math.floor((x0 - gx0) / BF))), min(gnx - 1, int(math.floor((x1 - gx0) / BF)))
            s, c = max(0, int(math.floor((y0 - gy0) / BF))), min(gny - 1, int(math.floor((y1 - gy0) / BF)))
            return [j * gnx + i for j in range(s, c + 1) for i in range(a, o + 1)]

        for k, bx in enumerate(pbox):
            for c in cells(*bx):
                gp[c].append(k)
        for k, row in enumerate(tr):
            x, y, _, trr, _, cx, cy, _, cr, _ = row
            for c in cells(min(cx - cr, x - trr), min(cy - cr, y - trr), max(cx + cr, x + trr), max(cy + cr, y + trr)):
                gt[c].append(k)

        def csr(lists):
            st = np.zeros(len(lists) + 1, np.int64)
            st[1:] = np.cumsum([len(q) for q in lists])
            return st, np.array([i for q in lists for i in q], np.int64)

        cps, cpi = csr(gp)
        cts, cti = csr(gt)
        self.grid = np.array([gx0, gy0, BF, gnx, gny], np.float64)
        self.rules = np.array([step_up, math.tan(math.radians(max_slope_deg)), r["canopyDensity"],
                               r["canopyBlockT"], 1.0 if TRUNK_COLLIDE else 0.0], np.float64)
        self.W = (H, self.tg, dil.ravel(), np.array([p["base"] for p in pr], float), np.array([p["top"] for p in pr], float),
                  np.array(pbox, float), np.array(pes, np.int64), np.array(edges, float), np.ascontiguousarray(tr),
                  self.grid, cps, cpi, cts, cti, np.zeros(len(pr), np.int64), np.zeros(len(tr), np.int64),
                  np.zeros(1, np.int64), self.rules)
        self.bounds = (b["minX"], b["maxX"], b["minY"], b["maxY"])

    # thin wrappers
    def ground(self, x, y):
        return _ground(self.W[0], self.tg, x, y)

    def surface(self, x, y, zmax=1e18):
        return _surface(self.W, x, y, zmax)

    def body_clear(self, x, y, z, radius, height):
        return _penetration(self.W, x, y, z, radius, height, False)[0] <= 1e-4

    def ray(self, a, b, trunks=False):
        """Fraction along a->b of the first solid hit, or NOHIT."""
        return _raycast(self.W, a[0], a[1], a[2], b[0], b[1], b[2], trunks)

    def sight(self, a, b):
        return _sight(self.W, a[0], a[1], a[2], b[0], b[1], b[2])

    def ray_many(self, A, B, trunks=False):
        return _ray_many(self.W, A, B, trunks)

    def sight_many(self, A, B, ignore_canopy=None):
        return _sight_many(self.W, A, B, ignore_canopy)

    def move_many(self, S):
        """S: (n, 9) rows x, y, z, radius, height, dx, dy, dz, grounded -> (n, 5) x, y, z, grounded, stoppedZ."""
        return _move_many(self.W, S)


# ---------------------------------------------------------------------- primitives
@njit(cache=True)
def _ground(H, tg, x, y):
    minX, minY, step, nx = tg[0], tg[1], tg[2], int(tg[3])
    ny = int(tg[4])
    r = (x - minX) / step
    i = (y - minY) / step
    r = 0.0 if r < 0 else (nx - 1.000001 if r > nx - 1.000001 else r)
    i = 0.0 if i < 0 else (ny - 1.000001 if i > ny - 1.000001 else i)
    a = int(r)
    o = int(i)
    s = r - a
    c = i - o
    l = o * nx + a
    return (H[l] * (1 - s) + H[l + 1] * s) * (1 - c) + (H[l + nx] * (1 - s) + H[l + nx + 1] * s) * c


@njit(cache=True)
def _gradient(H, tg, x, y):
    n = tg[2]
    return ((_ground(H, tg, x + n, y) - _ground(H, tg, x - n, y)) / (2 * n),
            (_ground(H, tg, x, y + n) - _ground(H, tg, x, y - n)) / (2 * n))


@njit(cache=True)
def _pip(ed, s, e, x, y):
    r = False
    for i in range(s, e):
        a, o, c_, d = ed[i, 0], ed[i, 1], ed[i, 2], ed[i, 3]
        if (o > y) != (d > y) and x < (c_ - a) * (y - o) / (d - o) + a:
            r = not r
    return r


@njit(cache=True)
def _nearest(ed, s, e, x, y):
    best, bx, by = 1e300, 0.0, 0.0
    for i in range(s, e):
        sx, sy = ed[i, 0], ed[i, 1]
        lx, ly = ed[i, 2] - sx, ed[i, 3] - sy
        dd = lx * lx + ly * ly
        f = ((x - sx) * lx + (y - sy) * ly) / dd if dd > XF else 0.0
        f = 0.0 if f < 0 else (1.0 if f > 1 else f)
        px, py = sx + lx * f, sy + ly * f
        h = (px - x) ** 2 + (py - y) ** 2
        if h < best:
            best, bx, by = h, px, py
    return bx, by, math.sqrt(best)


@njit(cache=True)
def _cell(gr, x, y):
    n = int(math.floor((x - gr[0]) / gr[2]))
    r = int(math.floor((y - gr[1]) / gr[2]))
    gnx, gny = int(gr[3]), int(gr[4])
    if n < 0 or r < 0 or n >= gnx or r >= gny:
        return -1
    return r * gnx + n


@njit(cache=True)
def _surface(W, x, y, zmax):
    H, tg, _, pb, pt, pbox, pes, ed, _, gr, cps, cpi = W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7], W[8], W[9], W[10], W[11]
    r = _ground(H, tg, x, y)
    c = _cell(gr, x, y)
    if c < 0:
        return r
    for k in range(cps[c], cps[c + 1]):
        p = cpi[k]
        if pt[p] > zmax or pt[p] <= r or x < pbox[p, 0] or x > pbox[p, 2] or y < pbox[p, 1] or y > pbox[p, 3]:
            continue
        if _pip(ed, pes[p, 0], pes[p, 1], x, y):
            r = pt[p]
    return r


@njit(cache=True)
def _contact_plane(W, x, y, rad, low, high, ceiling):
    """Closest ceiling or highest supporting prism under the body's footprint."""
    planes = W[3] if ceiling else W[4]
    box, pes, ed, gr, cps, cpi = W[5], W[6], W[7], W[9], W[10], W[11]
    best = 1e300 if ceiling else -1e300
    slip_x, slip_y = 0.0, 0.0
    for j in range(max(0, int(math.floor((y-rad-gr[1])/gr[2]))),
                   min(int(gr[4]), int(math.floor((y+rad-gr[1])/gr[2])) + 1)):
        for i in range(max(0, int(math.floor((x-rad-gr[0])/gr[2]))),
                       min(int(gr[3]), int(math.floor((x+rad-gr[0])/gr[2])) + 1)):
            cell = j * int(gr[3]) + i
            for k in range(cps[cell], cps[cell+1]):
                p = cpi[k]
                z = planes[p]
                if z < low or z > high or (z >= best if ceiling else z <= best):
                    continue
                if x+rad < box[p,0] or x-rad > box[p,2] or y+rad < box[p,1] or y-rad > box[p,3]:
                    continue
                inside = _pip(ed, pes[p,0], pes[p,1], x, y)
                cx, cy, d = _nearest(ed, pes[p,0], pes[p,1], x, y)
                if not inside and d >= rad:
                    continue
                best, slip_x, slip_y = z, 0.0, 0.0
                if not inside and d > XF:
                    slip = min(0.6, rad-d+0.001) / d
                    slip_x, slip_y = (x-cx)*slip, (y-cy)*slip
    return best, slip_x, slip_y


@njit(cache=True)
def _penetration(W, x, y, z, rad, height, allow_step=True):
    H, tg, pb, pt, pbox, pes, ed, tr, gr, cps, cpi, cts, cti, rules = (W[0], W[1], W[3], W[4], W[5], W[6], W[7], W[8],
                                                                       W[9], W[10], W[11], W[12], W[13], W[17])
    depth, bnx_, bny_ = 0.0, 0.0, 0.0
    a = z + (rules[0] if allow_step else 1e-6)
    o = z + height
    i0 = max(0, int(math.floor((x - rad - gr[0]) / gr[2])))
    i1 = min(int(gr[3]) - 1, int(math.floor((x + rad - gr[0]) / gr[2])))
    j0 = max(0, int(math.floor((y - rad - gr[1]) / gr[2])))
    j1 = min(int(gr[4]) - 1, int(math.floor((y + rad - gr[1]) / gr[2])))
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            cell = j * int(gr[3]) + i
            for k in range(cps[cell], cps[cell + 1]):
                p = cpi[k]
                if pt[p] <= a or pb[p] >= o or x + rad < pbox[p, 0] or x - rad > pbox[p, 2] or \
                        y + rad < pbox[p, 1] or y - rad > pbox[p, 3]:
                    continue
                cx, cy, d = _nearest(ed, pes[p, 0], pes[p, 1], x, y)
                if _pip(ed, pes[p, 0], pes[p, 1], x, y):
                    ax, ay = cx - x, cy - y
                    l = math.hypot(ax, ay)
                    if l == 0:
                        l = XF
                    if l + rad > depth:
                        depth, bnx_, bny_ = l + rad, ax / l, ay / l
                elif d < rad:
                    dd = d if d > 0 else XF
                    if rad - d > depth:
                        depth, bnx_, bny_ = rad - d, (x - cx) / dd, (y - cy) / dd
            if rules[4] > 0:
                for k in range(cts[cell], cts[cell + 1]):
                    t = cti[k]
                    if tr[t, 4] <= a or tr[t, 2] - 1 >= o:
                        continue
                    cx, cy = x - tr[t, 0], y - tr[t, 1]
                    u = math.hypot(cx, cy)
                    dd = rad + tr[t, 3]
                    if u < dd and dd - u > depth:
                        depth = dd - u
                        bnx_, bny_ = (cx / u, cy / u) if u > XF else (1.0, 0.0)
    if _ground(H, tg, x, y) > a:
        gx, gy = _gradient(H, tg, x, y)
        g = math.hypot(gx, gy)
        if g > 0.001:
            c_, l_ = -gx / g, -gy / g
            u = 0.0
            while u < 2 * rad and _ground(H, tg, x + c_ * u, y + l_ * u) > a:
                u += 0.25
            if max(u, 0.05) > depth:
                depth, bnx_, bny_ = max(u, 0.05), c_, l_
    for dv, nx_, ny_ in ((tg[7] + rad - x, 1.0, 0.0), (x + rad - tg[9], -1.0, 0.0),
                         (tg[8] + rad - y, 0.0, 1.0), (y + rad - tg[10], 0.0, -1.0)):
        if dv > depth:
            depth, bnx_, bny_ = dv, nx_, ny_
    return depth, bnx_, bny_


@njit(cache=True)
def _move(W, x, y, z, rad, height, dx, dy, dz, was_grounded):
    """moveAndSlide from the client, plus the protocol's slope limit."""
    H, tg, rules = W[0], W[1], W[17]
    step = rules[0]
    o, s, c = x, y, z
    u = max(1, int(math.ceil(math.hypot(dx, dy) / (rad * 0.4))))
    sx, sy = dx / u, dy / u
    for _ in range(u):
        px, py = o, s
        o += sx
        s += sy
        for _k in range(4):
            dep, nx_, ny_ = _penetration(W, o, s, c, rad, height)
            if dep <= 1e-4:
                break
            o += nx_ * (dep + 0.001)
            s += ny_ * (dep + 0.001)
        t = _surface(W, o, s, c + step)
        if t > c + 1e-3 and t <= _ground(H, tg, o, s) + 0.01:  # uphill on bare terrain
            gx, gy = _gradient(H, tg, o, s)
            if math.hypot(gx, gy) > rules[1]:
                o, s = px, py
                t = _surface(W, o, s, c + step)
        if t > c and dz <= 0:
            ce, _, _ = _contact_plane(W, o, s, rad, c+height-0.01, t+height-1e-6, True)
            if ce < 1e299:
                o, s = px, py
            else:
                c = t
    grounded, stopped = False, False
    h = _surface(W, o, s, c + step)
    g = c + dz
    if dz > 0:
        ce, ex, ey = _contact_plane(W, o, s, rad, c+height-0.01, g+height, True)
        if g + height > ce:
            if ex or ey:
                if _penetration(W, o+ex, s+ey, c, rad, height)[0] <= 1e-4:
                    o, s = o+ex, s+ey
                    ce, _, _ = _contact_plane(W, o, s, rad, c+height-0.01, g+height, True)
            if g + height > ce:
                g = ce - height
                stopped = True
    else:
        top, ex, ey = _contact_plane(W, o, s, rad, max(h, g)-1e-4, c+step, False)
        if top > h and g <= top+1e-4:
            if (ex or ey) and dx*ex + dy*ey >= 0:
                if _penetration(W, o+ex, s+ey, top, rad, height)[0] <= 1e-4:
                    o, s = o+ex, s+ey
                    h = _surface(W, o, s, c+step)
                    top, _, _ = _contact_plane(W, o, s, rad, max(h, g)-1e-4, c+step, False)
            h = max(top, h)
    if g <= h + 1e-4:
        g = h
        grounded = True
        stopped = stopped or dz < 0
    elif dz <= 0 and g - h <= 0.75 and was_grounded:
        g = h
        grounded = True
    # A step-height allowance during horizontal movement must not leave the
    # finished cylinder inside a solid, including at corners between prisms.
    for _ in range(12):
        dep, nx_, ny_ = _penetration(W, o, s, g, rad, height, False)
        if dep <= 1e-4:
            break
        o += nx_ * (dep + 0.001)
        s += ny_ * (dep + 0.001)
        g = max(g, _ground(H, tg, o, s))
    else:
        if _penetration(W, x, y, z, rad, height, False)[0] <= 1e-4:
            return x, y, z, was_grounded, True
    return o, s, g, grounded, stopped


@njit(cache=True)
def _move_many(W, S):
    out = np.empty((len(S), 5))
    for i in range(len(S)):
        r = _move(W, S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4], S[i, 5], S[i, 6], S[i, 7], S[i, 8] > 0.5)
        out[i, 0], out[i, 1], out[i, 2] = r[0], r[1], r[2]
        out[i, 3] = 1.0 if r[3] else 0.0
        out[i, 4] = 1.0 if r[4] else 0.0
    return out


# ---------------------------------------------------------------------- rays
@njit(cache=True)
def _slab(t0, d, lo, hi, l, u):
    if abs(d) < XF:
        return (t0 >= lo and t0 <= hi), l, u
    i = (lo - t0) / d
    a = (hi - t0) / d
    if i > a:
        i, a = a, i
    l = max(l, i)
    u = min(u, a)
    return l <= u, l, u


@njit(cache=True)
def _ray_prism(W, p, ox, oy, oz, dx, dy, dz, lim):
    pb, pt, pbox, pes, ed = W[3], W[4], W[5], W[6], W[7]
    l, u = 0.0, lim
    ok, l, u = _slab(ox, dx, pbox[p, 0], pbox[p, 2], l, u)
    if not ok:
        return NOHIT
    ok, l, u = _slab(oy, dy, pbox[p, 1], pbox[p, 3], l, u)
    if not ok:
        return NOHIT
    ok, l, u = _slab(oz, dz, pb[p], pt[p], l, u)
    if not ok:
        return NOHIT
    s, e = pes[p, 0], pes[p, 1]
    if oz >= pb[p] and oz <= pt[p] and _pip(ed, s, e, ox, oy):
        return 0.0
    best = lim
    hit = NOHIT
    for i in range(s, e):
        lx, ly = ed[i, 2] - ed[i, 0], ed[i, 3] - ed[i, 1]
        d = dx * ly - dy * lx
        if abs(d) < XF:
            continue
        hx, hy = ed[i, 0] - ox, ed[i, 1] - oy
        t = (hx * ly - hy * lx) / d
        v = (hx * dy - hy * dx) / d
        if t < 0 or t >= best or v < 0 or v > 1:
            continue
        zz = oz + dz * t
        if zz < pb[p] or zz > pt[p]:
            continue
        best = t
        hit = t
    if abs(dz) > XF:
        for k in range(2):
            plane = pt[p] if k == 0 else pb[p]
            t = (plane - oz) / dz
            if t < 0 or t >= best or (k == 0 and dz > 0) or (k == 1 and dz < 0):
                continue
            if _pip(ed, s, e, ox + dx * t, oy + dy * t):
                best = t
                hit = t
    return hit


@njit(cache=True)
def _ray_trunk(tr, t, ox, oy, oz, dx, dy, dz, lim):
    f, p = ox - tr[t, 0], oy - tr[t, 1]
    n = tr[t, 3]
    m = dx * dx + dy * dy
    if m < XF:
        return NOHIT
    h = 2 * (f * dx + p * dy)
    g = f * f + p * p - n * n
    disc = h * h - 4 * m * g
    if disc < 0:
        return NOHIT
    v = (-h - math.sqrt(disc)) / (2 * m)
    if v < 0 or v >= lim:
        return NOHIT
    zz = oz + dz * v
    if zz < tr[t, 2] - 1 or zz > tr[t, 4]:
        return NOHIT
    return v


@njit(cache=True)
def _ellipsoid(tr, t, ox, oy, oz, dx, dy, dz):
    """Entry and exit fractions of the segment through tree t's crown, or (NOHIT, NOHIT)."""
    cr, rz = tr[t, 8], tr[t, 9]
    s, c, l = (ox - tr[t, 5]) / cr, (oy - tr[t, 6]) / cr, (oz - tr[t, 7]) / rz
    u, d, f = dx / cr, dy / cr, dz / rz
    p = u * u + d * d + f * f
    if p < XF:
        return NOHIT, NOHIT
    m = 2 * (s * u + c * d + l * f)
    h = s * s + c * c + l * l - 1
    g = m * m - 4 * p * h
    if g <= 0:
        return NOHIT, NOHIT
    q = math.sqrt(g)
    v = (-m - q) / (2 * p)
    y = (-m + q) / (2 * p)
    if y < 0 or v > 1:
        return NOHIT, NOHIT
    return v, y


@njit(cache=True)
def _ray_terrain(W, ax, ay, az, bx, by, bz, lim):
    H, tg, bm = W[0], W[1], W[2]
    step, bnx, bny = tg[2], int(tg[5]), int(tg[6])
    c, l, u = bx - ax, by - ay, bz - az
    d = math.hypot(c, l)
    f = 8 * step
    if az - _ground(H, tg, ax, ay) < 0:
        return 0.0
    m = 0.5 * step / d if d > XF else 1.0
    h = 0.0
    it = 0
    while h < lim:
        it += 1
        if it > 100000:
            return NOHIT
        i, a = ax + c * h, ay + l * h
        g = int(math.floor((i - tg[0]) / f))
        q = int(math.floor((a - tg[1]) / f))
        v = g >= 0 and q >= 0 and g < bnx and q < bny
        y = lim
        if d > XF:
            e, t = tg[0] + g * f, tg[1] + q * f
            n = (e + f - i) / c if c > XF else ((e - i) / c if c < -XF else 1e300)
            r = (t + f - a) / l if l > XF else ((t - a) / l if l < -XF else 1e300)
            y = min(h + max(0.0, min(n, r)) + 1e-6, lim)
        b = min(az + u * h, az + u * y)
        if v and b > bm[q * bnx + g]:
            h = y
            continue
        x = min(h + m, lim)
        if az + u * x - _ground(H, tg, ax + c * x, ay + l * x) < 0:
            lo, hi = h, x
            for _ in range(20):
                mid = (lo + hi) / 2
                if az + u * mid - _ground(H, tg, ax + c * mid, ay + l * mid) < 0:
                    hi = mid
                else:
                    lo = mid
            return hi
        if x >= lim:
            break
        h = x
    return NOHIT


@njit(cache=True)
def _walk(gr, ax, ay, bx, by, cells):
    """Grid cells crossed by a->b in order (client walkCells). Returns the count written."""
    bf, gx0, gy0, gnx, gny = gr[2], gr[0], gr[1], int(gr[3]), int(gr[4])
    a, o = bx - ax, by - ay
    s = int(math.floor((ax - gx0) / bf))
    c = int(math.floor((ay - gy0) / bf))
    l = int(math.floor((bx - gx0) / bf))
    u = int(math.floor((by - gy0) / bf))
    d = 1 if a > 0 else -1
    f = 1 if o > 0 else -1
    p = bf / abs(a) if abs(a) > XF else 1e300
    m = bf / abs(o) if abs(o) > XF else 1e300
    h = ((s + 1) * bf + gx0 - ax if a > 0 else ax - (s * bf + gx0)) / abs(a) if abs(a) > XF else 1e300
    g = ((c + 1) * bf + gy0 - ay if o > 0 else ay - (c * bf + gy0)) / abs(o) if abs(o) > XF else 1e300
    n = 0
    for _ in range(4096):
        if s >= 0 and c >= 0 and s < gnx and c < gny:
            cells[n] = c * gnx + s
            n += 1
        if s == l and c == u:
            break
        if h < g:
            t = h
            h += p
            s += d
        else:
            t = g
            g += m
            c += f
        if t > 1:
            break
    return n


@njit(cache=True)
def _raycast(W, ax, ay, az, bx, by, bz, trunks):
    tr, gr, cps, cpi, cts, cti, sP, sT, sc = W[8], W[9], W[10], W[11], W[12], W[13], W[14], W[15], W[16]
    dx, dy, dz = bx - ax, by - ay, bz - az
    sc[0] += 1
    st = sc[0]
    cells = np.empty(4096, np.int64)
    n = _walk(gr, ax, ay, bx, by, cells)
    best = 1.0
    hit = False
    for q in range(n):
        cell = cells[q]
        for k in range(cps[cell], cps[cell + 1]):
            p = cpi[k]
            if sP[p] == st:
                continue
            sP[p] = st
            t = _ray_prism(W, p, ax, ay, az, dx, dy, dz, best)
            if t < best or (t == 0.0 and not hit):
                best, hit = t, True
        if trunks:
            for k in range(cts[cell], cts[cell + 1]):
                t_ = cti[k]
                if sT[t_] == st:
                    continue
                sT[t_] = st
                t = _ray_trunk(tr, t_, ax, ay, az, dx, dy, dz, best)
                if t < best:
                    best, hit = t, True
    t = _ray_terrain(W, ax, ay, az, bx, by, bz, best)
    if t < best:
        best, hit = t, True
    return best if hit else NOHIT


@njit(cache=True)
def _canopy(W, ax, ay, az, bx, by, bz):
    """Metres of the segment inside tree crowns, summed per tree (client canopyTransmittance)."""
    tr, gr, cts, cti, sT, sc = W[8], W[9], W[12], W[13], W[15], W[16]
    dx, dy, dz = bx - ax, by - ay, bz - az
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    sc[0] += 1
    st = sc[0]
    cells = np.empty(4096, np.int64)
    n = _walk(gr, ax, ay, bx, by, cells)
    u = 0.0
    for q in range(n):
        cell = cells[q]
        for k in range(cts[cell], cts[cell + 1]):
            t = cti[k]
            if sT[t] == st:
                continue
            sT[t] = st
            v, y = _ellipsoid(tr, t, ax, ay, az, dx, dy, dz)
            if v != NOHIT:
                u += (min(y, 1.0) - max(v, 0.0)) * L
    return u


@njit(cache=True)
def _sight(W, ax, ay, az, bx, by, bz, ignore_canopy=False):
    rules = W[17]
    if _raycast(W, ax, ay, az, bx, by, bz, True) != NOHIT:
        return False
    return ignore_canopy or math.exp(-rules[2] * _canopy(W, ax, ay, az, bx, by, bz)) >= rules[3]


@njit(cache=True)
def _sight_many(W, A, B, ignore_canopy):
    out = np.empty(len(A), np.bool_)
    for i in range(len(A)):
        skip = ignore_canopy is not None and ignore_canopy[i]
        out[i] = _sight(W, A[i, 0], A[i, 1], A[i, 2], B[i, 0], B[i, 1], B[i, 2], skip)
    return out


@njit(cache=True)
def _ray_many(W, A, B, trunks):
    out = np.empty(len(A))
    for i in range(len(A)):
        out[i] = _raycast(W, A[i, 0], A[i, 1], A[i, 2], B[i, 0], B[i, 1], B[i, 2], trunks)
    return out
