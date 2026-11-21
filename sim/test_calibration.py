"""Regression checks for mechanics calibrated against the kit stream and protocol."""
import math

from engine import MAP_DIR, Match, Mech, Thing, get_geom, load_config


def fixture():
    match = Match.__new__(Match)
    match.cfg = cfg = load_config()
    match.G = get_geom(MAP_DIR, cfg)
    match.t = 0.0
    match.sounds = []
    match.mechs = [Mech(0, 0, 0, 0, 0, cfg, {}, None)]
    return match, match.mechs[0]


def test_jet_takeoff_and_sustained_climb():
    match, m = fixture()
    # Remove collisions to isolate the movement integrator.
    match._slide = lambda bodies, dz: None
    m.move, m.jump, m.boost, m.energy = (1, 0), True, True, 29.4
    for expected_energy in (27.9, 26.4):
        match._move()
        match._move()
        assert math.isclose(m.vz, 8.9)
        assert math.isclose(m.energy, expected_energy)


def test_rear_hit_event():
    match, victim = fixture()
    shooter = Mech(1, -10, 0, 0, 0, match.cfg, {}, None)
    for x, rear, amount in ((-10, True, 67.5), (10, False, 45)):
        match._damage(victim, 45, shooter, "gun", (x, 0))
        for event in (victim.events[-1], shooter.events[-1]):
            assert event["amount"] == amount
            assert event.get("rear", False) == rear


def test_proximity_sight():
    match, observer = fixture()
    match.projs, match.cubes, match.pickups, match.crates = [], [], [], []
    # A dense crown in the kit arena. Both target rays pass above the trunk.
    x, y, eye_z = -418.92, 286.06, 12.46
    target = Thing(id=1, x=x+15, y=y, z=eye_z-4.5, h=9)
    item = Thing(id=2, x=target.x, y=y, z=eye_z)
    match.projs.append(item)
    observer.y, observer.z = y, eye_z - observer.stats["eye_height"]

    def check(visible):
        assert (target.id in match._visibility([observer], [target])[observer.id]) == visible
        assert (("projectile", item.id) in match._item_visibility([observer])[observer.id]) == visible

    observer.x, observer.yaw = x-15, math.pi
    check(True)   # 30 m, behind the observer, through canopy
    observer.x, observer.yaw = x-35, 0
    check(False)  # 50 m, inside the wedge, same crown
    # A real tree trunk occludes both samples in proximity range.
    observer.x, observer.y, observer.z = 510.75, 122.25, 10
    target.x, target.y, target.z = 530.75, 122.25, 13.5
    item.x, item.y, item.z = target.x, target.y, 18
    check(False)


    # A building blocks proximity sight across its short axis (about 24 m).
    observer.x, observer.y, observer.z = -112.2, -215.7, -5.64
    target.x, target.y, target.z = -112.2, -192, -2.14
    item.x, item.y, item.z = target.x, target.y, 2.36
    check(False)


class Scripted:
    def __init__(self, hello):
        self.reply = {}
        self.views = []

    def act(self, obs):
        self.views.append(obs)
        if self.reply is None:
            return None
        return {"tick": obs["tick"], **self.reply}


def arena(count=2):
    """A collision-free test space above the actual map, with production actor methods."""
    g = Match(load_config(), 12, [Scripted] * 16)
    g.mechs = g.mechs[:count]
    g.crates, g.npcs, g.pickups = [], [], []
    for i, m in enumerate(g.mechs):
        m.x, m.y, m.z, m.yaw = i * 20.0, 0.0, 200.0, (math.pi if i else 0.0)
        m.cyaw = m.yaw
    return g


def equip(g, m, kind, slot=1):
    g._add_pickup(kind, m.x, m.y, m.z)
    g._take(m, g.pickups[-1].id, slot)
    w = next(w for w in m.slots if w and w["kind"] == kind)
    w["equip"] = 0.0
    return w


def test_reply_coast():
    g = arena(1)
    m = g.mechs[0]
    m.bot.reply = {"move": [1, 0], "boost": True, "aim": [2, 1], "fire": [True, False], "jump": True}
    g.decide()
    assert m.fire[0] and m.jump
    for tick in range(2, 23, 2):
        g.tick, g.t = tick, tick / 20
        m.bot.reply = None if tick < 10 else {"tick": 0, "fire": [True, True]}
        g.decide()
        assert not any(m.fire) and not m.jump
        assert m.move == ((1, 0) if tick <= 20 else (0, 0))
        assert m.boost == (tick <= 20)
    assert (m.cyaw, m.cpitch) == (m.yaw, m.pitch)
    m.bot.reply = {"move": [0, 1]}
    g.decide()
    assert m.move == (0, 1)


def test_observed_slot_shape():
    g = arena(1)
    m = g.mechs[0]
    w = equip(g, m, "laser")
    w["equip"] = 1.65
    w["cooldown"] = -0.01
    g.decide()
    gun, laser = m.bot.views[-1]["self"]["weapons"]
    assert set(gun) == {"kind", "ammo", "reload", "cooldown", "stats"}
    assert set(laser) == {"kind", "charge", "cooldown", "stats"}
    assert laser["cooldown"] == 1.65


def test_lockstep_contested_take_and_events():
    g = arena()
    a, b = g.mechs
    a.x, b.x = 3.8, 1
    g._add_pickup("laser", 0, 0, a.z)
    pid = g.pickups[-1].id
    a.bot.reply = b.bot.reply = {"take": pid}
    g.decide()
    assert a.kinds() == ["gun"] and b.kinds() == ["gun", "laser"]
    assert [e["type"] for e in b.events] == ["took"]
    assert a.events[-1]["reason"] == "gone"
    assert all(any(p["id"] == pid for p in m.bot.views[-1]["pickups"]) for m in (a,b))
    a.bot.reply, b.bot.reply = {}, {"drop": 1}
    g.decide()
    assert b.bot.views[-1]["events"][-1]["type"] == "took"
    assert b.events[-1] == {"type":"dropped", "kind":"laser", "slot":1, "id":g.pickups[-1].id}
    g.decide()
    assert b.bot.views[-1]["events"][-1]["type"] == "dropped"


def test_simultaneous_death_and_no_revival():
    g = arena(3)
    a,b,c = g.mechs
    g._give_cubes(b, 3)
    g._damage(b, 5000, a, "gun", (0,0))
    g._damage(b, 5000, c, "laser", (0,0))
    g._spawn_cube(b.x,b.y,b.z)
    g._cubes_pickup()
    assert b.cubes == 3
    g._zone_regen()
    c.hp = -1
    g._deaths()
    assert (b.place,c.place)==(3,3) and a.kills==1 and b.hp==c.hp==0
    assert len(g.cubes)==1+5+2
    g.decide()
    assert not b.bot.views and not c.bot.views
    assert g.run(max_t=.05)["place"] == [1,3,3]


def test_cross_kind_visibility():
    g = arena(1)
    m = g.mechs[0]
    g._add_pickup("laser",m.x+5,m.y,m.z)
    p = g._spawn_proj("gun",m,1,0,0,252,45,230,.2,7)
    p.id, p.x = g.pickups[0].id,-100
    visible=g._item_visibility([m])[m.id]
    assert ("pickup",p.id) in visible and ("projectile",p.id) not in visible


def test_beam_visibility():
    g=arena(1)
    m=g.mechs[0]
    for coords,visible in [([(-100,0),(-80,0)],False), ([(190,-210),(190,210)],True), ([(201,-5),(201,5)],False)]:
        path=[[x,y,m.z+6] for x,y in coords]
        g.beams=[(99,"laser",path,0)]
        g.decide()
        assert bool(m.bot.views[-1]["beams"])==visible


def test_fire_cadence_and_reload():
    g=arena(1)
    m=g.mechs[0]
    gun=m.slots[0]
    m.bot.reply={"fire":[True,False]}
    g.decide()
    ticks=[]
    for i in range(140):
        start=len(g.projs)
        g._weapons()
        if len(g.projs)>start:ticks.append(i)
    assert ticks[:12]==[0,5,10,15,20,24,29,34,39,44,48,53]
    assert ticks[12]-ticks[11]==48
    assert min(b-a for a,b in zip(ticks[12:],ticks[13:]))>=4
    m.fire=[False,False]
    gun.update(ammo=3,reload=0)
    m.reload_req=True
    g._weapons()
    m.reload_req=False
    assert gun["reload"]==2.4 and g.sounds[-1][0]=="reload"
    for _ in range(48):g._weapons()
    assert gun["ammo"]==12 and gun["reload"]==0


def test_gun_final_segment_and_barrel_wall():
    g=arena()
    shooter,target=g.mechs
    p=g._spawn_proj("gun",shooter,1,0,0,252,45,230,.1,7)
    target.x=p.x+10
    p.dist=228
    g._projectiles()
    assert not g.projs and target.hp==target.max_hp
    shooter.x,shooter.y,shooter.z=0,0,200
    p=g._spawn_proj("gun",shooter,0.6,0,0.8,252,45,230,.1,7)
    for _ in range(19):g._projectiles()
    assert not g.projs
    # A thin wall cuts the segment between the mech and the nominal barrel tip.
    g.G=type("Wall",(),{"ray":lambda self,a,b: .5})()
    assert math.isclose(g._barrel_offset(shooter,7,1,0,0),1.5,abs_tol=.00002)


def test_laser_release_curve_and_effects():
    g=arena(3)
    m,a,b=g.mechs
    w=equip(g,m,"laser")
    equip(g,a,"mortar")
    g._upgrade(m,"laser_pierce")
    m.fire[1]=True
    for _ in range(8):g._weapons()
    assert m.charging() and g.sounds[-1][0]=="laser_charge"
    g.decide()
    assert a.bot.views[-1]["mechs"][0]["charging"]
    g._weapons()
    expected=315*(.35+.65*.4/.75)
    assert math.isclose(a.max_hp-a.hp,expected*1.6)
    assert math.isclose(b.max_hp-b.hp,expected)
    assert a.events[-2]["effect"]=="cookoff" and w["cooldown"]==1.65
    assert g.sounds[-1][0]=="laser"
    g=arena()
    m,target=g.mechs
    w=equip(g,m,"laser")
    target.x=285
    m.fire[1]=True
    for _ in range(55):g._weapons()
    assert w["charge"]==0 and target.hp==target.max_hp
    assert math.isclose(math.dist(*g.beams[-1][2]),280)


def test_mortar_ballistics_and_concussion():
    g=arena(4)
    owner,target,edge,out=g.mechs
    ws=owner.wst["mortar"]
    equip(g,owner,"mortar")
    owner.fire[1]=True
    g._weapons()
    p=g.projs[0]
    assert math.isclose(math.hypot(p.vx,p.vy,p.vz),72) and p.g==22
    assert owner.slots[1]["cooldown"]==2.6 and g.sounds[-1][0]=="mortar"
    g._projectiles()
    assert p.vz==-1.1
    owner.x,target.x,edge.x,out.x=0,1,9.5,9.6
    for m in g.mechs:m.yaw=math.pi
    owner.yaw=0
    g._explode(p,0,0,owner.z+4,target)
    assert target.max_hp-target.hp==360 and math.isclose(edge.max_hp-edge.hp,48)
    assert owner.max_hp-owner.hp==80 and out.hp==out.max_hp
    assert target.energy==55 and (target.vx,target.vz)==(16,9)
    assert any(e.get("effect")=="concuss" for e in owner.events if e["type"]=="hit")
    assert any(e["type"]=="effect" and e["effect"]=="concuss" for e in target.events)
    g.G=type("Wall",(),{"ray":lambda self,a,b: .5})()
    before=edge.hp
    g._explode(p,0,0,owner.z+4,None)
    assert edge.hp==before


def test_lightning_3d_chain_and_short():
    g=arena(5)
    m,a,b,c,high=g.mechs
    a.x,b.x,c.x,high.x=10,23,36,11
    high.z+=50
    for q in (a,b,c,high):q.yaw=math.pi
    equip(g,m,"lightning")
    laser=equip(g,a,"laser")
    laser["charge"]=.4
    a.fire[1]=m.fire[1]=True
    g._weapons()
    assert [round(q.max_hp-q.hp,1) for q in (a,b,c,high)]==[60,42,29.4,0]
    assert m.energy==97 and m.slots[1]["cooldown"]==.25
    assert laser["charge"]==0 and a.short_until==2
    m.fire[1]=False
    g.t=1.95
    g._weapons()
    assert laser["charge"]==0
    g.t=2
    g._weapons()
    assert laser["charge"]>0
    for q in (a,b,c):q.x=-20
    m.slots[1]["cooldown"]=0
    m.fire[1]=True
    g._weapons()
    assert m.energy==94 and math.isclose(math.dist(*g.beams[-1][2]),45)


def test_drones_return_and_cadence():
    g=Match(load_config(),1,[Scripted]*16)
    n=g.npcs[0]
    g.npcs,g.mechs=[n],g.mechs[:1]
    n.hx,n.hy,n.x,n.y,n.z=0,0,0,0,200
    m=g.mechs[0]
    m.x,m.y,m.z=40,0,200
    for i in range(11):
        g.t=i/10
        g._npcs_think()
    assert n.target is m and len(g.projs)==2
    p=g.projs[-1]
    assert p.dmg==25 and math.isclose(math.hypot(p.vx,p.vy,p.vz),90)
    n.x,m.x=61,70
    g._npcs_think()
    assert n.target is None and (n.gx,n.gy)==(0,0)
    assert n.hp==350
    g._damage(n,400,m,"gun",(m.x,m.y))
    assert not n.alive and len(g.cubes)==2


def test_supply_clock():
    g=arena()
    m,b=g.mechs
    for t in (70,140,210):
        g.t=t
        g._drops_step()
        d=g.drops[-1]
        phase=g.zone_at(t)[2]
        assert d.lands_at==t+25 and math.dist((d.x,d.y),phase[4])<=phase[5]
    d=g.drops[0]
    g.drops=[d]
    m.x,m.y,m.z=d.x+8,d.y,d.z
    b.x=d.x+20
    g.t=94.95
    g._drops_step()
    assert d.progress==0
    g.t=95
    for _ in range(30):g._drops_step()
    before=d.progress
    b.x,b.y,b.z=d.x,d.y,d.z
    g._drops_step()
    assert d.progress==before and d.holder==m.id
    m.x=d.x+10
    g._drops_step()
    assert d.holder==b.id and d.progress==.05
    b.x=d.x+20
    g._drops_step()
    assert d.holder is None and d.progress==0
    m.x,m.energy,m.hp=d.x,2,100
    for _ in range(79):g._drops_step()
    assert d in g.drops and not m.cubes
    g._drops_step()
    assert not g.drops and (m.cubes,m.hp,m.energy)==(3,m.max_hp,100)


def test_jet_sound_and_steps():
    g=arena(1)
    m=g.mechs[0]
    g._slide=lambda *args:None
    m.jump=m.boost=True
    g._move()
    assert g.sounds[-1][0]=="jet"
    g._move()
    assert len(g.sounds)==1
    m.boost=False
    m.grounded=True
    for speed,sound in [(4,None),(11.7,"step"),(19.5,"boost")]:
        m.boost=sound=="boost"
        m.move=(speed/(19.5 if m.boost else 11.7),0)
        m.vx=speed
        m.step_acc=0
        g.sounds=[]
        for _ in range(12):g._move()
        assert [e[0] for e in g.sounds]==([sound] if sound else [])


def test_progression_and_offers():
    g=arena(1)
    m=g.mechs[0]
    m.hp-=100
    for i in range(1,46):
        prior=m.pending
        g._give_cubes(m,1)
        assert m.pending-prior==int(i in (1,3,6,10,15,21,28,36,45))
        assert math.isclose(m.max_hp,1400*(1+.04*min(i,10)))
        assert math.isclose(m.hp,m.max_hp-100)
        assert math.isclose(m.dmg_mult,1+.04*min(i,10))
        for key in ("radius","height","eye_height","muzzle_height"):
            assert math.isclose(m.stats[key],g.cfg["mech"][key]*(1+.05*i))
    g._offers()
    u=m.offer["choices"][0]["id"]
    allowed={x["id"] for x in g.cfg["upgrades"] if x.get("requires") in (None,"gun")}
    ids={x["id"] for x in m.offer["choices"]}
    assert len(ids)==3 and ids<=allowed
    g.t=4.99
    g._offers()
    assert not m.upgrades
    g.t=5
    g._offers()
    assert m.upgrades==[u] and m.offer["id"]==2
    for u in g.cfg["upgrades"]:
        fresh=Mech(0,0,0,0,0,g.cfg,g.upg,None)
        for stack in range(1,u["max"]+1):
            g._upgrade(fresh,u["id"])
            for e in u["effects"]:
                path=e["path"].split(".")
                base=g.cfg["mech"] if path[0]=="mech" else g.cfg["weapons"][path[1]]
                stats=fresh.stats if path[0]=="mech" else fresh.wst[path[1]]
                exp=base[path[-1]]*e.get("mul",1)**stack+stack*e.get("add",0)
                assert math.isclose(stats[path[-1]],exp), (u["id"],stack)


def test_house_and_tournament_seats():
    from unittest.mock import patch
    import house
    from collections import Counter
    assert Counter(house.LEVELS)==dict(easy=4,medium=4,hard=4,expert=3)
    games=[]
    def end(g):
        games.append(g)
        return {"place":list(range(1,17))}
    cfg=load_config()
    with patch.object(Match,"run",end):
        pts,res=house.practice(cfg,4,{})
        g=games[-1]
        assert pts==16-res["place"][g.track] and len(g.levels)==15
        pl=g.mechs[g.track]
        g._update_shared(g.mechs,{1:{pl.id}})
        assert g.shared["last"]==(0,(pl.x,pl.y,pl.z))
        h=g.shared["hunters"]
        assert len(h)==1 and g.levels[next(iter(h))]==0
        squad=[m for m in g.mechs if m.team==1]
        assert all(m not in g._targets(squad[0]) for m in squad)
        g._update_shared([m for m in g.mechs if m.id not in h],{})
        assert len(g.shared["hunters"])==2
        g.t=240
        g._update_shared(g.mechs,{})
        assert len(g.shared["hunters"])==15
        points,res=house.ffa(cfg,3,[{},{}])
        g=games[-1]
        assert len(points)==2 and sum(isinstance(m.bot,house.HouseBot) for m in g.mechs)==14
        assert all(m.team is None for m in g.mechs)



def test_serializable_observation_after_moving():
    import json
    g=arena(2)
    g._move()
    g.decide()
    for m in g.mechs:
        view=m.bot.views[-1]
        assert type(view["self"]["grounded"]) is bool
        assert json.loads(json.dumps(view,allow_nan=False))["tick"]==g.tick


def test_sounds_ranges_noise_buckets_and_age():
    from statistics import pstdev
    g=arena(1)
    m=g.mechs[0]
    for kind,limit in g.cfg["sound"]["range"].items():
        g.sounds=[(kind,limit,0,7,1.9),(kind,limit+.01,0,7,1.9),(kind,1,0,m.id,1.9)]
        g.t=2
        g.decide()
        sounds=m.bot.views[-1]["sounds"]
        assert len(sounds)==1 and sounds[0]["kind"]==kind and sounds[0]["age"]==.1
    g.sounds=[("gun",x,0,8,2) for x in (59.99,60,159.99,160)]
    g.decide()
    assert [s["dist"] for s in m.bot.views[-1]["sounds"]]==["near","mid","mid","far"]
    g.sounds=[("gun",100,0,8,2)]*4000
    g.decide()
    bearings=[s["bearing"] for s in m.bot.views[-1]["sounds"]]
    assert abs(math.degrees(pstdev(bearings))-7)<.3


def test_memory_updates_and_boundary():
    g=arena(2)
    m,b=g.mechs
    g.decide()
    assert m.bot.views[-1]["mechs"][0]["id"]==b.id
    b.x=-150
    for tick in (98,100,102):
        g.tick,g.t=tick,tick*.05
        b.x-=1
        g.decide()
        enemies=m.bot.views[-1]["mechs"]
        if tick<=100:assert enemies[0]["pos"]==[b.x,b.y,b.z]
        else:assert not enemies


def test_health_energy_pitch_and_zone_damage():
    g=arena(1)
    m=g.mechs[0]
    g._apply(m,{"aim":[2,20],"move":[3,4]})
    assert m.move==(.6,.8) and m.cpitch==1.3
    g._slide=lambda *args:None
    m.last_spend=0
    m.energy=40
    g.t=.55
    g._move()
    assert m.energy==40 and math.isclose(m.pitch,.15)
    g.t=.6
    g._move()
    assert math.isclose(m.energy,40.9) and math.isclose(m.pitch,.30)
    m.last_dmg=0
    m.hp=100
    g.t=4.95
    g._zone_regen()
    assert m.hp==100
    g.t=5
    g._zone_regen()
    assert m.hp==102.25
    m.x=10000
    for ph,dps in zip(g.phases,[20,35,60,100,160,260]):
        for g.t in (ph[0]-.05,ph[0],ph[1]-.05):
            hp=m.hp
            g._zone_regen()
            assert math.isclose(hp-m.hp,dps*.05) and not m.in_zone
            assert m.last_dmg==g.t


def test_loot_failures_crates_size_and_drops():
    g=Match(load_config(),5,[Scripted]*16)
    assert len(g.crates)==36 and len(g.pickups)==40 and len(g.npcs)==12
    assert all(c.hp==300 for c in g.crates)
    assert all(w.kind in ("mortar","lightning","laser") for w in g.pickups)
    assert len({w.kind for w in g.pickups})==3
    # Deterministic sides of the 50% roll, through crate destruction.
    from unittest.mock import patch
    m=g.mechs[0]
    for i,roll in enumerate((.499,.501)):
        c=g.crates[i]
        nc,np=len(g.cubes),len(g.pickups)
        with patch.object(g.rng,"random",return_value=roll):
            g._damage(c,400,m,"gun",(0,0))
        assert len(g.cubes)==nc+1 and len(g.pickups)==np+(roll<.5)
    g=arena()
    m,b=g.mechs
    def take(kind,x=0,z=0,drop=None):
        g._add_pickup(kind,m.x+x,m.y,m.z+z)
        g._take(m,g.pickups[-1].id,drop)
        return m.events[-1]
    assert take("laser",4.001)["reason"]=="range"
    assert take("laser",z=4.001)["reason"]=="range"
    assert take("gun")["reason"]=="have"
    assert take("laser",4,4)["type"]=="took"
    assert take("mortar")["reason"]=="full"
    m.slots[1]["charge"]=.2
    assert take("mortar",drop=1)["reason"]=="busy"
    g._apply(m,{"drop":1})
    assert m.slots[1]["kind"]=="laser"
    m.slots[1]["charge"]=0
    assert take("mortar",drop=1)["type"]=="took"
    g._take(m,-1,None)
    assert m.events[-1]["reason"]=="gone"
    m.slots[0]["ammo"]=4
    m.slots[0]["reload"]=1.4
    g._apply(m,{"drop":0})
    dropped=g.pickups[-1]
    g._take(m,dropped.id,None)
    assert m.slots[0]["ammo"]==4 and m.slots[0]["reload"]==1.4 and m.slots[0]["equip"]==1
    g._upgrade(m,"mortar_reload")
    g._apply(m,{"drop":1})
    pk=g.pickups[-1]
    g._take(m,pk.id,None)
    assert math.isclose(m.wst["mortar"]["cooldown"],2.6*.82)
    m.hp=0
    count=len(g.pickups)
    g._deaths()
    assert len(g.pickups)==count+1 and g.pickups[-1].kind=="mortar"
    g=arena()
    m,b=g.mechs
    b.x,b.y=10,3.5
    for cubes in (0,12):
        g._give_cubes(b,cubes)
        p=g._spawn_proj("gun",m,1,0,0,252,45,230,.1,7)
        hp=b.hp
        g._projectiles()
        assert (b.hp<hp)==bool(cubes)


def test_temporal_and_weapon_boundaries():
    g=arena(1)
    m=g.mechs[0]
    m.bot.reply={"fire":[True,True]}
    g.decide()
    for _ in range(2):
        g.step()
        assert m.fire==[True,True]
    m.bot.reply={}
    g.decide()
    assert m.fire==[False,False]
    g=arena(2)
    m,b=g.mechs
    w=equip(g,m,"laser")
    m.fire[1]=True
    g._weapons()
    m.fire[1]=False
    g._weapons()
    assert not g.beams
    g._weapons()
    assert len(g.beams)==1 and math.isclose(b.max_hp-b.hp,315*(.35+.65*.15/.75))
    w["cooldown"]=0
    m.fire[1]=True
    for _ in range(15):g._weapons()
    m.move=(1,0)
    g._slide=lambda *args:None
    for _ in range(10):g._move()
    assert math.isclose(m.vx,11.7*.5)
    m.fire[1]=False
    b.hp=b.max_hp
    g._weapons()
    assert b.max_hp-b.hp==315
    g=arena(3)
    m,a,b=g.mechs
    equip(g,m,"lightning")
    m.fire[1]=True
    a.x,a.y,b.x,b.y=13,4,53,0
    g._weapons()
    assert a.hp==a.max_hp and b.hp==b.max_hp
    first=g.beams[-1][2][0]
    assert first==[3,0,m.z+7]
    a.x,a.y=43,0
    b.x=25
    m.slots[1]["cooldown"]=0
    g._weapons()
    assert b.hp<b.max_hp and a.hp==a.max_hp
    class Wall:
        def ray(self,a,b,trunks=False):
            return .5 if b[0]>10 and a[0]<10 else 2
    g.G=Wall()
    m.slots[1]["cooldown"]=0
    hp=b.hp
    g._weapons()
    assert b.hp==hp


def test_cube_proximity_and_offer_stack_limits():
    g=arena(2)
    a,b=g.mechs
    for x in (3.5,3.51):
        g.cubes=[Thing(id=5000,x=x,y=0,z=a.z+4)]
        before=a.cubes
        g._cubes_pickup()
        assert a.cubes-before==int(x==3.5)
    g.cubes=[Thing(id=5000,x=0,y=0,z=a.z+4.01)]
    g._cubes_pickup()
    assert g.cubes
    a.x,b.x=-1,1
    g.mechs=[b,a]
    g.cubes=[Thing(id=5000,x=0,y=0,z=a.z)]
    g._cubes_pickup()
    assert b.cubes==0 and a.cubes==2
    a.pending=1
    for u in g.cfg["upgrades"]:
        if u["id"]!="armor":a.upc[u["id"]]=u["max"]
    g._offers()
    assert [c["id"] for c in a.offer["choices"]]==["armor"]
    g._apply(a,{"pick":0})
    assert a.upgrades==["armor"] and a.offer is None


def test_isolated_geometry():
    import json, tempfile
    from pathlib import Path
    import numpy as np
    from geom import Geom
    with tempfile.TemporaryDirectory() as td:
        v=Path(td)/"v2"
        v.mkdir()
        world=json.loads((Path(MAP_DIR)/"v2/world.json").read_text())
        world["bounds"]={"minX":0,"minY":0,"maxX":60,"maxY":60}
        world["terrain"].update(minX=0,minY=0,nx=61,ny=61,step=1,file="floor.i16")
        world["trees"]["data"]=[]
        def prism(x,y,base,top,width=8):
            return {"base":base,"top":top,"ring":[x,y,x+width,y,x+width,y+8,x,y+8]}
        world["prisms"]=[prism(20,20,10,12),prism(35,20,0,18),prism(20,40,0,1.2)]
        (v/"world.json").write_text(json.dumps(world))
        terrain=np.zeros((61,61),dtype="<i2")
        terrain[51:,5:10]=np.arange(5)*200
        terrain.tofile(v/"floor.i16")
        G=Geom(td)
        def move(x,y,z,dx=0,dy=0,dz=0,h=9,rad=2.5):
            return G.move_many(np.array([[x,y,z,rad,h,dx,dy,dz,0]],float))[0]
        assert move(3,10,0,-4)[0]>=2.5-1e-5
        assert move(19.8,44,0,1)[2]>=1.2
        assert move(31,24,0,5,1)[0]<33 and move(31,24,0,5,1)[1]>24
        assert move(6,55,2,.5,rad=.2)[0]==6
        assert move(21,24,0,2)[2]==0 and G.ray((19,24,7),(31,24,7))>1
        assert move(24,24,0,dz=3)[2]==1
        eave=move(19,24,0,dz=3)
        assert math.isclose(eave[0],18.4) and eave[2]==1
        roof=move(19,24,12.2,dz=-.5)
        assert math.isclose(roof[0],18.4) and roof[2]==12
        # Two overlapped mechs pinned against a building resolve inside the walkable region.
        g=arena(2);g.G=G
        a,b=g.mechs
        a.x,b.x=31.5,32.4
        for m in (a,b):m.y,m.z=24,0
        g._move()
        assert b.x<=32.501 and a.x<b.x



def test_slot_seed_equip_delay_and_held_charge():
    class Hello(Scripted):
        def __init__(self,h):
            super().__init__(h)
            self.hello=h
    matches=[Match(load_config(),seed,[Hello]*16) for seed in (3,4)]
    hello=[m.bot.hello for m in matches[0].mechs]
    assert [h["slot"] for h in hello]==list(range(16))
    assert len({h["seed"] for h in hello})==16
    assert [h["seed"] for h in hello]==[m.bot.hello["seed"] for m in matches[1].mechs]
    assert all(h["protocol"]=="mechfight/1" and h["map"]["files"]==["v2/world.json","v2/terrain.i16"] for h in hello)
    assert matches[0].mechs[0].x!=matches[1].mechs[0].x
    g=arena(1)
    m=g.mechs[0]
    g._add_pickup("laser",0,0,m.z)
    g._take(m,g.pickups[-1].id,None)
    m.fire[1]=True
    for _ in range(33):
        g._weapons()
        assert not m.charging()
    for _ in range(17):g._weapons()
    assert m.charging() and g._weapon_obs(m,m.slots[1])["charge"]==.75
    m.slots[1]["charge"]=0
    m.fire[1]=False
    m.bot.reply={"debug":{"label":"test"}}
    g.decide()
    assert m.move==(0,0) and not any(m.fire)


def test_crate_top_and_growth_next_to_wall():
    import numpy as np
    g=arena(1)
    m=g.mechs[0]
    g.crates=[Thing(id=1000,kind="crate",x=20,y=0,z=m.z,h=4,alive=True,hp=300)]
    # Opaque cover hides the center while exposing the top.
    class Cover:
        def sight_many(self,eyes,points,ignore_canopy):
            return points[:,2]>m.z+3.9
    geom=g.G
    g.G=Cover()
    assert ("crate",1000) in g._item_visibility([m])[m.id]
    g.G=geom
    m.x,m.y,m.z=-68.0917511164,-198.5510226974,1.4501635427
    m.grounded=False
    g._give_cubes(m,1)
    from geom import _penetration
    assert _penetration(geom.W,m.x,m.y,m.z,m.r,m.h,False)[0]<=.001
    assert np.isfinite([m.x,m.y,m.z]).all()


def test_house_shared_vision_and_friendly_damage():
    g=arena(3)
    player,spotter,ally=g.mechs
    player.x,spotter.x,ally.x=0,30,350
    spotter.team=ally.team=1
    g.decide()
    for bot in (spotter,ally):
        assert [m["id"] for m in bot.bot.views[-1]["mechs"]]==[player.id]
    g._damage(spotter,100,ally,"gun",(ally.x,ally.y))
    assert spotter.hp==spotter.max_hp and not spotter.events


def test_drone_acquisition_and_deaggro():
    g=Match(load_config(),4,[Scripted]*16)
    n,m=g.npcs[0],g.mechs[0]
    g.npcs,g.mechs=[n],[m]
    n.hx,n.hy,n.x,n.y,n.z=0,0,30,0,200
    m.x,m.y,m.z=75.01,0,200
    g._npcs_think()
    assert n.target is None
    m.x=75
    g._npcs_think()
    assert n.target is m
    m.x=121
    g._npcs_think()
    assert n.target is None
    n.x,m.x=30.1,32
    g._npcs_think()
    assert n.target is None
    g._damage(n,1,m,"gun",(m.x,m.y))
    assert n.target is m
    hp=n.hp
    g.t=417
    g._zone_regen()
    assert n.hp==hp


def test_zone_interpolation_and_timeout_order():
    g=arena(3)
    for start,end,c0,r0,c1,r1,_ in g.phases:
        assert math.dist(c0,c1)+r1<=r0+1e-8
        center,radius,_=g.zone_at((start+end)/2)
        assert math.isclose(radius,(r0+r1)/2)
        assert math.dist(center,[(a+b)/2 for a,b in zip(c0,c1)])<1e-8
    g.t,g.tick=479.95,9599
    for i,m in enumerate(g.mechs):m.hp=500+100*i
    r=g.run()
    assert r["place"]==[3,2,1] and r["t"]==480



def test_spawn_clearance_floor_distribution_and_persistence():
    import itertools
    for seed in (2,6):
        g=Match(load_config(),seed,[Scripted]*16)
        assert all(math.hypot(a.x-b.x,a.y-b.y)>=130 for a,b in itertools.combinations(g.mechs,2))
        assert all(g.G.body_clear(m.x,m.y,m.z,m.r,m.h) for m in g.mechs)
        assert sum(c.z>g.G.ground(c.x,c.y)+.05 for c in g.crates)==9
        assert sum(p.z>g.G.ground(p.x,p.y)+.05 for p in g.pickups)==10
        ids={p.id for p in g.pickups}
        g.t=416
        g._zone_regen()
        g._drops_step()
        assert {p.id for p in g.pickups}==ids


def test_mortar_interior_falloff_and_scaled_impulse():
    g=arena(2)
    m,b=g.mechs
    m.x=-10
    b.x=6
    equip(g,m,"mortar")
    m.fire[1]=True
    g._weapons()
    p=g.projs[0]
    g._explode(p,0,0,b.z+4,None)
    # Nearest cylinder surface is 3.5 m from the blast, halfway to the edge.
    assert math.isclose(b.max_hp-b.hp,104)
    assert math.isclose(b.vx,10.4) and math.isclose(b.vz,5.85)
    assert math.isclose(b.energy,70.75)


def test_damage_regen_armor_and_energy_cap():
    g=arena(2)
    a,b=g.mechs
    for m in (a,b):m.hp=600
    g._upgrade(b,"armor")
    g.t=1
    g._damage(b,100,a,"gun",(a.x,a.y))
    assert math.isclose(b.hp,508) and a.last_dmg==b.last_dmg==1
    g.t=5.95
    g._zone_regen()
    assert a.hp==600 and b.hp==508
    g.t=6
    g._zone_regen()
    assert (a.hp,b.hp)==(602.25,510.25)
    b.x=10000
    g._zone_regen()
    assert b.hp==509.25
    b.x=a.x+20
    g.t=10.95
    g._zone_regen()
    assert b.hp==509.25
    g.t=11
    g._zone_regen()
    assert b.hp==511.5
    a.energy=99.9
    g._move()
    assert a.energy==100


def test_geometry_bridge_overhang_and_terrain_layout():
    import json
    from pathlib import Path
    import numpy as np
    g,m=fixture()
    world=json.loads((Path(MAP_DIR)/"v2/world.json").read_text())
    t=world["terrain"]
    grid=np.fromfile(Path(MAP_DIR)/"v2"/t["file"],dtype="<i2").reshape(724,1070)
    x,y,dx,dy=120,70,.2,.7
    h=t["scale"]*((1-dy)*((1-dx)*grid[y,x]+dx*grid[y,x+1])+dy*((1-dx)*grid[y+1,x]+dx*grid[y+1,x+1]))
    assert math.isclose(g.G.ground(t["minX"]+x+dx,t["minY"]+y+dy),h,abs_tol=1e-8)



def test_protocol_observation_examples_and_actions():
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(MAP_DIR).parent/"check"))
    from check import Schema
    schema=Schema(str(Path(MAP_DIR).parent/"mechfight.schema.json"))
    g=arena(2)
    a,b=g.mechs
    g.crates=[Thing(id=1000,kind="crate",alive=True,hp=300,h=2.5,x=12,y=0,z=a.z)]
    g.cubes=[Thing(id=5000,x=14,y=0,z=a.z)]
    g._add_pickup("laser",17,0,a.z)
    own=g._spawn_proj("gun",a,1,0,0,252,45,230,.1,7)
    foreign=g._spawn_proj("gun",b,-1,0,0,252,45,230,.1,7)
    g.t,g.tick=70,1400
    g._drops_step()
    a.pending=1
    g._offers()
    g.decide()
    o=a.bot.views[-1]
    assert not schema.errors(o,"Observation")
    assert o["alive"]==2 and o["tick"]==1400 and o["t"]==70
    assert o["self"]["weapons"][1] is None and o["offer"]["id"]==1
    assert [x["id"] for x in o["pickups"]]==[g.pickups[0].id] and g.pickups[0].id>=8000
    assert [x["id"] for x in o["cubes"]]==[5000] and o["crates"][0]["hp"]==300
    assert [x["id"] for x in o["projectiles"]]==[foreign.id] and own.id!=foreign.id
    assert o["drops"][0]["id"]==20000 and o["drops"][0]["lands_in"]==25
    assert all(m.bot.views[-1]["drops"]==o["drops"] for m in (a,b))
    assert o["zone"]["next_radius"]==380
    g._apply(a,{"aim":[math.pi/2,-4],"move":[0,-.3],"extra":"ignored"})
    assert a.cpitch==-.7 and a.move==(0,-.3)
    a.yaw,a.pitch=math.pi/2,math.pi/6
    dx,dy,dz=g._dir(a)
    assert abs(dx)<1e-8 and math.isclose(dy,math.sqrt(3)/2) and math.isclose(dz,.5)
    assert all(math.isclose(a.stats[k],v) for k,v in dict(radius=2.5,height=9,eye_height=8,muzzle_height=7).items())
    assert a.hp==1400
    assert not schema.errors({"type":"end","place":1,"kills":0},"EndMsg")
    assert json.loads(json.dumps(o))["type"]=="obs"


def test_drone_cover_and_sound_sources():
    g=Match(load_config(),4,[Scripted]*16)
    n,m=g.npcs[0],g.mechs[0]
    g.npcs,g.mechs=[n],[m]
    n.hx,n.hy,n.x,n.y,n.z=0,0,0,0,200
    m.x,m.y,m.z=30,0,200
    class Cover:
        def sight(self,a,b):
            return False
        def ray(self,a,b,trunks=False):
            return .5
    g.G=Cover()
    g._npcs_think()
    assert n.target is None and not g.projs
    n.target=m
    g._npcs_think()
    assert n.target is m and not g.projs
    g=arena(2)
    a,b=g.mechs
    c=Thing(kind="crate",x=0,y=0,z=200,hp=10,alive=True,id=1000)
    g._damage(c,11,a,"gun",(0,0))
    assert g.sounds[-1][0]=="crate"
    g._damage(b,2000,a,"gun",(0,0))
    g._deaths()
    assert g.sounds[-1][0]=="death" and a.events[-1]=={"type":"kill","target":b.id}
    for m in g.mechs:m.x=m.y=0
    g.t=70
    g._drops_step()
    g.t=95
    g._drops_step()
    assert g.sounds[-1][0]=="drop"


def test_spawned_item_and_npc_shapes():
    g=Match(load_config(),5,[Scripted]*16)
    m=g.mechs[0]
    n=g.npcs[0]
    m.x=m.y=0
    m.z=200
    n.x,n.y,n.z=10,0,200
    n.yaw=.5
    g.mechs=[m]
    g.npcs=[n]
    g.decide()
    npc=m.bot.views[-1]["mechs"][0]
    assert npc["kind"]=="npc" and npc["id"]>=100 and npc["hp"]==350
    assert set(npc)=={"kind","id","pos","vel","aim","hp","max_hp"}
    assert n.r==1.8 and n.h==3.5



if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_"):
            check()
            print(name, "passed")
