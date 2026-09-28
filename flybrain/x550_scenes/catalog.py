"""Procedural static obstacle families; no downloaded assets or hidden collisions."""
import numpy as np

FAMILIES=('warehouse','doorway','scaffold','woodland','utility_yard','shelving')
PALETTES={'wood':(.40,.25,.12),'metal':(.40,.43,.46),
          'concrete':(.55,.53,.48),'paint':(.75,.48,.12),'foliage':(.22,.36,.18)}


def make_scene(rng, family=None):
    family=family or FAMILIES[int(rng.integers(len(FAMILIES)))]
    if family not in FAMILIES: raise ValueError('unknown scene family')
    objects=[]
    shift=float(rng.uniform(-.25,.25)); forward=float(rng.uniform(-.3,.3))
    def box(name,n,e,z,size,material='wood',yaw=0.):
        objects.append(dict(name=name,shape='box',north=n+forward,east=e+shift,
            altitude=z,size=list(size),yaw=float(yaw),material=material))
    def pole(name,n,e,r=.12,h=4.6,material='metal'):
        objects.append(dict(name=name,shape='cylinder',north=n+forward,east=e+shift,
            altitude=h/2,size=[2*r,2*r,h],yaw=0.,material=material))
    if family=='warehouse':
        for j,(n,e) in enumerate(((3.,-1.1),(5.5,1.1),(8.,-.65))):
            h=float(rng.uniform(2.7,3.8))
            box('stacked_crate_'+str(j),n,e,h/2,(1.1,1.1,h),yaw=rng.uniform(-.3,.3))
        pole('support_column',6.5,-2.5,.22)
    elif family=='doorway':
        gap=float(rng.uniform(1.55,2.25)); top=float(rng.uniform(2.8,3.65))
        for e in (-gap/2-.12,gap/2+.12):
            box('door_post_'+str(len(objects)),4.,e,2.,(.35,.24,4.),'concrete')
        box('door_lintel',4.,0.,top,(.35,gap+.48,.22),'concrete')
        box('cabinet',7.,1.3,1.7,(1.1,.75,3.4),'metal')
    elif family=='scaffold':
        for j,(n,e) in enumerate(((3.5,-1.2),(3.5,1.2),(6.5,-1.2),(6.5,1.2))):
            pole('scaffold_post_'+str(j),n,e,.075,4.4)
        box('crossbar_front',3.5,0.,3.25,(.12,2.55,.14),'paint')
        box('crossbar_rear',6.5,0.,2.55,(.12,2.55,.14),'paint')
        box('side_rail',5.,1.2,3.6,(3.,.12,.12),'metal')
    elif family=='woodland':
        for j,(n,e) in enumerate(((2.8,-1.2),(4.2,.9),(5.8,-.5),(7.4,1.4),(8.7,-1.7))):
            pole('tree_trunk_'+str(j),n,e,float(rng.uniform(.14,.32)),4.8,'wood')
        box('low_branch',5.8,-.1,2.7,(.15,1.6,.16),'wood',rng.uniform(-.4,.4))
        box('high_branch',7.4,.9,3.8,(.15,1.5,.14),'wood',rng.uniform(-.4,.4))
    elif family=='utility_yard':
        pole('tank',4.,-.9,.65,3.5)
        pole('utility_pole',6.,1.4,.17,4.8)
        box('sign_arm',6.,.7,3.25,(.15,1.55,.14),'metal')
        box('barrier',8.,-.5,1.2,(.45,2.3,2.4),'paint')
        box('equipment_box',3.,2.,1.55,(.8,.7,3.1),'metal')
    elif family=='shelving':
        for j,(n,e) in enumerate(((3.,-1.25),(3.,1.25),(6.,-1.25),(6.,1.25))):
            pole('rack_upright_'+str(j),n,e,.07,4.2)
        box('upper_shelf',4.5,0.,3.7,(3.15,2.65,.14),'wood')
        box('lower_shelf',4.5,0.,2.0,(3.15,2.65,.14),'wood')
        box('stored_box',4.5,.65,2.65,(.8,.8,1.15),'wood')
    assert 0<len(objects)<=8
    return dict(schema='x550-scene-v1',family=family,objects=objects,
                goal_north=11.,goal_east=float(rng.uniform(-1.,1.)))
