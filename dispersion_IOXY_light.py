import argparse
from math import pi, sqrt, sin, cos
import random
import numpy as np
import vtk
import signal
from collections import namedtuple

import espressomd.magnetostatics as magnetostatics
from espressomd.interactions import FeneBond
from espressomd import system
from espressomd.virtual_sites import VirtualSitesRelative
from espressomd import checkpointing

from multiprocessing import Process, Queue, Manager

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lam", type=int, help="Lambda", default=2)
    parser.add_argument("-fl", "--fl",
                        type=int, help="Filament len", default=10)
    parser.add_argument("-fn", "--fn",
                        type=int, help="Filament number", default=1)
    parser.add_argument("-phi", "--dens",
                        type=float, help="dens", default=0.05)
    parser.add_argument("-c", "--conf",
                        type=int, help="1 - chain, 2 - ring, 3 - X, 4 - Y", default=1)
    parser.add_argument("-kf", "--k_fene",
                        type=int, help="Fene arg", default=100)
    parser.add_argument("--seed",
                        type=int, help="seed for random numbers", default=5)
    cmd_args = parser.parse_args()

    return cmd_args


def add_virtual_particle(s, vtype, sys_topology, x, y, z, rpid):
    vpid = len(s.part)
    s.part.add(id=vpid, type=vtype, pos=[x, y, z])
    s.part[vpid].virtual = True
    s.part[vpid].vs_auto_relate_to(rpid)
    sys_topology.vpids.append(vpid)
    return vpid


def add_bond_between(p_one, p_two, s, bond, sys_topology):
    s.part[p_one].add_bond((bond, s.part[p_two]))
    sys_topology.connected_virtuals.append([p_two, p_one])
    sys_topology.connected_reals.append(
        [s.part[p_two].vs_relative[0], s.part[p_one].vs_relative[0]])


def generateChains(s, ext_params, particle_params, bond):
    fn, fl = ext_params.fn, ext_params.fl
    mom = particle_params.mom
    rtype, vtype = particle_params.rtype, particle_params.vtype
    sys_topology = SysTopology([], [], [], [])

    rand1 = random.random()
    rand2 = random.random()
    pi = 3.14

    sin_theta = -1.0 + 2 * rand1
    cos_theta = sqrt(1 - sin_theta * sin_theta)
    cos_phi = cos(2 * pi * rand2)
    sin_phi = sin(2 * pi * rand2)
    bond_length = particle_params.bond_length
    direction = np.array([cos_theta * cos_phi, cos_theta * sin_phi, sin_theta])

    radius = particle_params.sigma * 0.5
    for i in range(fn):
        x, y, z = s.box_l[0] * np.random.random(3)

        for j in range(fl):
            rpid = len(s.part)
            s.part.add(id=rpid, type=rtype,
                       pos=[x, y, z], dip=mom * direction, rotation=(1, 1, 1))
            sys_topology.rpids.append(rpid)

            if j > 0:
                vpid = add_virtual_particle(s, vtype, sys_topology,
                                            x - radius * direction[0],
                                            y - radius * direction[1],
                                            z - radius * direction[2], rpid)
                add_bond_between(vpid, vpid - 2, s, bond, sys_topology)

            if j < fl - 1:
                vpid = add_virtual_particle(s, vtype, sys_topology,
                                            x + radius * direction[0],
                                            y + radius * direction[1],
                                            z + radius * direction[2], rpid)

            x += bond_length * direction[0]
            y += bond_length * direction[1]
            z += bond_length * direction[2]

    return sys_topology


def generateRings(s, ext_params, particle_params, bond):
    fn, fl = ext_params.fn, ext_params.fl
    mom = particle_params.mom
    rtype, vtype = particle_params.rtype, particle_params.vtype
    sys_topology = SysTopology([], [], [], [])

    R = bond_length / (2 * sin(pi / fl))
    dTheta = 2 * pi / fl
    radius = particle_params.sigma * 0.5
    centers = []

    for i in range(fn):
        print('Generate ', i, ' filament')
        x0, y0, z0 = s.box_l[0] * np.random.random(3)

        first_virt_pid = 0
        distance_checked = False

        if len(centers) == 0:
            centers.append([x0, y0, z0])
        else:
            while not distance_checked:
                if check_centers(x0, y0, z0, centers, R):
                    x0, y0, z0 = s.box_l[0] * np.random.random(3)
                else:
                    distance_checked = True
                    centers.append([x0, y0, z0])

        for j in range(fl):
            theta = j * dTheta

            m_x = -mom * sin(theta)
            m_y = mom * cos(theta)
            m_z = 0

            xr = x0 + R * cos(theta)
            yr = y0 + R * sin(theta)
            zr = z0

            rpid = len(s.part)
            s.part.add(id=rpid, type=rtype,
                       pos=[xr, yr, zr], dip=[m_x, m_y, m_z],
                       rotation=(1, 1, 1))
            sys_topology.rpids.append(rpid)

            vpid = add_virtual_particle(s, vtype, sys_topology, xr - radius * m_x,
                                        yr - radius * m_y, z0, rpid)

            if j == 0:
                print('Ring beging from ', vpid - 1)
                first_virt_pid = vpid

            vpid = add_virtual_particle(s, vtype, sys_topology, xr - radius * m_x,
                                        yr - radius * m_y, z0, rpid)

            if j == 0:
                continue

            if j != fl - 1:
                add_bond_between(vpid - 1, vpid - 3, s, bond, sys_topology)

            if j == fl - 1:
                add_bond_between(vpid, first_virt_pid, s, bond, sys_topology)

    return sys_topology


def create_branch(s, particle_params, sys_topology, bond, init_rpid, init_coords, direction, branch_len, inverse_dir_factor=1):
    x, y, z = init_coords
    direction = direction / np.linalg.norm(direction)
    vpid_prev = len(sys_topology.vpids) - 1
    mom = particle_params.mom
    radius = 0.5*particle_params.sigma

    vx, vy, vz = np.array([x,y,z]) + radius * direction
    vpid_prev = add_virtual_particle(s, vtype, sys_topology, vx, vy, vz, init_rpid)

    for j in range(branch_len):
        x += bond_length * direction[0]
        y += bond_length * direction[1]
        z += bond_length * direction[2]

        rpid = len(s.part)
        m_factor = mom * inverse_dir_factor
        s.part.add(id=rpid, type=rtype,
                   pos=[x, y, z], dip=[m_factor * direction[0], m_factor * direction[1], 0],
                   rotation=(1, 1, 1))

        sys_topology.rpids.append(rpid)

        vx, vy, vz = np.array([x,y,z]) - radius * direction
        vpid = add_virtual_particle(s, vtype, sys_topology,
                                    vx, vy, vz, rpid)

        add_bond_between(vpid_prev, vpid, s, bond, sys_topology)

        if j < branch_len - 1:
            vx, vy, vz = np.array([x,y,z]) + radius * direction
            vpid_prev = add_virtual_particle(s, vtype, sys_topology,
                                             vx, vy, vz, rpid)


def generateX(s, ext_params, particle_params, bond):
    fn, fl = ext_params.fn, ext_params.fl
    mom = particle_params.mom
    rtype, vtype = particle_params.rtype, particle_params.vtype
    sys_topology = SysTopology([], [], [], [])
    bond_length = particle_params.bond_length

    branch_len = (fl - 1) / 4
    radius = particle_params.sigma * 0.5

    if branch_len.is_integer():
        branch_len = int(branch_len)
    else:
        raise ValueError("Please input a filament length that makes (fl - 1)/4 an integer number")

    for i in range(fn):
        x0, y0, z0 = s.box_l[0] * np.random.random(3)

        rpid = len(s.part)
        s.part.add(id=rpid, type=rtype, pos=[x0, y0, z0], dip=[-mom, 0, 0])
        sys_topology.rpids.append(rpid)

        create_branch(s, particle_params, sys_topology, bond,
                      rpid, [x0, y0, z0], [-radius, -radius, 0], branch_len)

        create_branch(s, particle_params, sys_topology, bond,
                      rpid, [x0, y0, z0], [-radius, +radius, 0], branch_len)

        create_branch(s, particle_params, sys_topology, bond,
                      rpid, [x0, y0, z0], [+radius, -radius, 0], branch_len, inverse_dir_factor=-1)

        create_branch(s, particle_params, sys_topology, bond,
                      rpid, [x0, y0, z0], [+radius, +radius, 0], branch_len, inverse_dir_factor=-1)



    return sys_topology


def generateY(s, ext_params, particle_params, bond):
    fn, fl = ext_params.fn, ext_params.fl
    mom = particle_params.mom
    rtype, vtype = particle_params.rtype, particle_params.vtype
    radius = particle_params.sigma * 0.5
    sys_topology = SysTopology([], [], [], [])

    branch_len = (fl - 1) / 3
    if branch_len.is_integer():
        branch_len = int(branch_len)
    else:
        print("Please input a filament length that makes (fl - 1)/3 an integer number")

    for i in range(fn):
        x0, y0, z0 = s.box_l[0] * np.random.random(3)

        rpid = len(s.part)
        s.part.add(id=rpid, type=rtype, pos=[x0, y0, z0], rotation=(1, 1, 1), dip=[-mom, 0, 0])
        sys_topology.rpids.append(rpid)

        create_branch(s, particle_params, sys_topology, bond,
                      rpid, [x0, y0, z0], [radius, 0, 0], branch_len, inverse_dir_factor=-1)

        create_branch(s, particle_params, sys_topology, bond,
                      rpid, [x0, y0, z0], [-radius, radius, 0], branch_len)

        create_branch(s, particle_params, sys_topology, bond,
                      rpid, [x0, y0, z0], [-radius, -radius, 0], branch_len)
    return sys_topology


def check_centers(x0, y0, z0, centers, r):
    for i in range(len(centers)):
        if too_close(x0, y0, z0, centers[i], r):
            return True
    return False


def too_close(x0, y0, z0, center, r):
    return centers_close(x0, y0, z0, center, r) and coords_close(
        x0, y0, z0, center, r)


def centers_close(x0, y0, z0, center, r):
    return sqrt(
        abs(x0 - center[0])**2 + abs(y0 - center[1])**2 +
        abs(z0 - center[2])**2) <= 2 * r


def coords_close(x0, y0, z0, center, r):
    return abs(x0 - center[0]) < 2 * r or abs(y0 - center[1]) < 2 * r or abs(
        z0 - center[2]) < 2 * r


def write_dip_pos_file(args): #s, fname, rtype, time, idx_to_write):
    #====
    s = args["s"]
    fname = args["fname"]
    rtype = args["rtype"]
    time = args["time"]
    idx_to_write = args["idx_to_write"]
    #===

    fname = fname + '.step.{}.dat'.format(time)
    ids = np.array(idx_to_write)
    particles = s.part[ids]
    pos = particles.pos
    dip = particles.dip
    type_part = particles.type
    np.savetxt(
        fname,
        np.column_stack([ids, type_part, pos, dip]),
        fmt=['%d', '%d', '%0.4f', '%0.4f', '%0.4f', '%0.4f', '%0.4f', '%0.4f'],
        header='pid type posx posy posz mx my mz')


def write_vtk_file(args):#s, fname, time, real_part_ids, fl, bonds=[]
    #===
    s = args["s"]
    fname = args["fname"]
    time = args["time"]
    real_part_ids = args["real_part_ids"]
    fl = args["fl"]
    bonds=[]
    if "bonds" in args:
        bonds = args["bonds"]
    #===

    fname = fname + '.step.{}.vtk'.format(time)
    point_num = len(s.part[:])
    part_idx = list(np.arange(point_num))

    if len(real_part_ids) > 0:
        part_idx = real_part_ids
        point_num = len(real_part_ids)
        # print(part_idx)

    ugrid_l = vtk.vtkUnstructuredGrid()

    # first of all trasfer points to vtk structure
    points = vtk.vtkPoints()
    for i in part_idx:
        px, py, pz = s.part[i].pos_folded
        points.InsertNextPoint(px, py, pz)

    ugrid_l.SetPoints(points)

    moments = vtk.vtkFloatArray()
    moments.SetNumberOfComponents(3)
    moments.SetName('mag_moments')
    for i in part_idx:
        mx, my, mz = s.part[i].dip
        moments.InsertNextTuple3(mx, my, mz)

    ugrid_l.GetPointData().SetVectors(moments)

    if bonds:
        cellArray_l = vtk.vtkCellArray()
        for pair in bonds:
            # print('pair in bonds', part_idx.index(pair[0]), part_idx.index(pair[1]))
            line_vtk = vtk.vtkLine()
            line_vtk.GetPointIds().SetId(0, part_idx.index(pair[0]))
            line_vtk.GetPointIds().SetId(1, part_idx.index(pair[1]))
            cellArray_l.InsertNextCell(line_vtk)
        ugrid_l.SetCells(vtk.VTK_LINE, cellArray_l)

    mask = vtk.vtkFloatArray()
    mask.SetNumberOfComponents(1)
    mask.SetName('cluster_num')
    marker = 0
    for n in range(point_num):
        mask.InsertNextValue(marker)
        if (n + 1) % fl == 0:
            marker += 1
    ugrid_l.GetPointData().AddArray(mask)

    mask_num = vtk.vtkFloatArray()
    mask_num.SetNumberOfComponents(1)
    mask_num.SetName('particle_num')
    for n in part_idx:
        if s.part[n].virtual:
            mask_num.InsertNextValue(-n)
        else:
            mask_num.InsertNextValue(n)
    ugrid_l.GetPointData().AddArray(mask_num)

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(ugrid_l)
    writer.Write()


def raise_finish_flag(args):
    #===
    filename = args["filename"]
    data = args["data"]
    #===
    with open(filename, 'w') as flag_file:
        flag_file.write(data)

def f(q,b):
    while (b.value or (not q.empty())):
        while (not q.empty()):
            data = q.get()
            func = data["func"]
            args = data["args"]
            func(args)
        time.sleep(1)

if __name__ == "__main__":

    #writing process init

    wq = Queue()
    check = Manager().Value('i',1)
    p = Process(target=f, args=(q,check))
    p.start()

    #default logic

    cmd_args_dict = read_args().__dict__
    ExtParams = namedtuple('external_params', cmd_args_dict.keys())
    ext_params = ExtParams(*cmd_args_dict.values())
    print(ext_params)

    exmperiment_code = f'{ext_params.conf}.'
    for p in ext_params._fields:
        if p != 'conf':
            exmperiment_code += f'{p}.{getattr(ext_params, p)}.'
    print('exmperiment_code:', exmperiment_code)

    FINISH_FLAG_NAME = 'is_continue_t{}_fn_{}_k_{}_seed{}.txt'.format(
        ext_params.conf, ext_params.fn, ext_params.k_fene, ext_params.seed)

    # Path("./data").mkdir(parents=True, exist_ok=True)
    box_l = pow(ext_params.fl * ext_params.fn / ext_params.dens, 1. / 3.)
    # box_l = 200
    print("box_l:", box_l)

    rtype = 0
    vtype = 1

    _sigma = 1.0
    _epsilon = 1.0
    _epsilon_attr = 1.0
    _r_cut = 2**(1 / 6)
    _r_cut_attr = 2.5 * _sigma
    _mom = sqrt(ext_params.lam)
    _d_r = 2 * _sigma
    bond_length = 1.2

    SysTopology = namedtuple('system_topology_list', [
        'rpids', 'vpids', 'connected_reals', 'connected_virtuals'])
    sys_topology = SysTopology([], [], [], [])
    state = 'no_state'
    steps = 0
    main_steps = 0
    main_run_steps = 400
    equilibrium_steps = 1000
    steps_to_record_from = 50


    checkpoint_id = exmperiment_code
    checkpoint_id = checkpoint_id.replace('.', '_')
    checkpoint = checkpointing.Checkpoint(checkpoint_id=checkpoint_id,
                                          checkpoint_path=".")

    vtf_file_name = exmperiment_code + ".vtf"
    dippos_file_name = exmperiment_code + ".dat"

    sim_sys = system.System(box_l=[box_l, box_l, box_l])

    sim_sys.seed = ext_params.seed
    sim_sys.time_step = 0.00005
    sim_sys.cell_system.skin = 0.3
    sim_sys.periodicity = [0, 0, 0]
    sim_sys.virtual_sites = VirtualSitesRelative(have_velocity=True)
    energy_data = np.zeros(equilibrium_steps)

    # here we store parameters that are needed in other functions
    TransferedParticleInfo = namedtuple('Particle_params',
                                        'mom sigma rtype vtype  bond_length')
    particle_params = TransferedParticleInfo(_mom, _sigma, rtype, vtype,
                                             bond_length)

    if checkpoint.has_checkpoints():
        sys_topology_dict = {}
        checkpoint.load()
        print('Checkpoint is loaded')
        for v in [
                'state', 'steps', 'main_steps', 'sim_sys.time_step',
                'sim_sys.periodicity'
        ]:
            print(v + ':', eval(v))
        if sim_sys.actors.active_actors:
            print('sim_sys.actors.active_actors', sim_sys.actors.active_actors)

        sys_topology = SysTopology(*sys_topology_dict.values())
        print(sys_topology)
        print("Continue")
    else:
        sim_sys.non_bonded_inter[rtype, rtype].lennard_jones.set_params(
            epsilon=_epsilon, sigma=_sigma, cutoff=_r_cut, shift="auto")

        fene_bond = FeneBond(k=ext_params.k_fene, d_r_max=_d_r)
        sim_sys.bonded_inter.add(fene_bond)
        # sim_sys.min_global_cut = 1.5 * _sigma # there is no in tcl

        if ext_params.conf == 1:
            sys_topology = generateChains(sim_sys, ext_params, particle_params, fene_bond)

        if ext_params.conf == 2:
            sys_topology = generateRings(sim_sys, ext_params, particle_params, fene_bond)

        if ext_params.conf == 3:
            sys_topology = generateX(sim_sys, ext_params, particle_params, fene_bond)

        if ext_params.conf == 4:
            sys_topology = generateY(sim_sys, ext_params, particle_params, fene_bond)

        sys_topology_dict = sys_topology._asdict()
        checkpoint.register('sim_sys', 'state', 'steps', 'main_steps',
                            'sys_topology_dict')


        print('len(reals):', len(sys_topology.rpids))
        print('real:', sys_topology.rpids)
        print('virt:', sys_topology.vpids)
        print('connected_reals:', sys_topology.connected_reals)
        print('connected_virtuals:', sys_topology.connected_virtuals)

        #s, fname, time, real_part_ids, fl, bonds=[]

        wq.put({"func": write_vtk_file,"args": {"s": sim_sys, "fname": ("init_" + dippos_file_name),
                       "time": 0, "real_part_ids": [], "fl": ext_params.fl,
                       "bonds": sys_topology.connected_reals}})

        #write_vtk_file({"s": sim_sys, "fname": ("init_" + dippos_file_name),
        #               "time": 0, "real_part_ids": [], "fl": ext_params.fl,
        #               "bonds": sys_topology.connected_reals})
        state = "warm_up"

        print("Start")

    if state == "warm_up":
        sim_sys.thermostat.set_langevin(kT=4, gamma=1, seed=ext_params.seed)
        sim_sys.time_step = 0.00005
        sim_sys.periodicity = [1, 1, 1]
        # To obtain the initial forces one has to initialize the integrator
        sim_sys.integrator.run(0)

        min_dist = 0.8
        min_dist_times = 1000
        act_min_dist = sim_sys.analysis.min_dist([rtype], [rtype])
        cap = 20
        sim_sys.force_cap = cap

        i = 0
        while i < min_dist_times and act_min_dist < min_dist:
            sim_sys.integrator.run(100)
            act_min_dist = sim_sys.analysis.min_dist([rtype], [rtype])
            i += 1
            cap = cap + 10
            sim_sys.force_cap = cap
            print(act_min_dist, sim_sys.analysis.min_dist())
            if i % 1 == 0:
                #s, fname, time, real_part_ids, fl, bonds=[]
                wq.put({"func": write_vtk_file,"args": {"s": sim_sys, "fname": ("warmup_" + dippos_file_name),
                               "time": i, "real_part_ids": sys_topology.rpids, "fl": ext_params.fl,
                               "bonds": sys_topology.connected_reals}})
                #write_vtk_file({"s": sim_sys, "fname": ("warmup_" + dippos_file_name),
                #               "time": i, "real_part_ids": sys_topology.rpids, "fl": ext_params.fl,
                #               "bonds": sys_topology.connected_reals})
        print("i = {} from {} for warm up".format(i, min_dist_times))
        sim_sys.force_cap = 0
        state = "shake_loop"

    if state == "shake_loop":
        sim_sys.thermostat.set_langevin(kT=1, gamma=1, seed=ext_params.seed)
        sim_sys.time_step = 0.00005
        try:
            for i in range(steps, equilibrium_steps):
                sim_sys.integrator.run(100)
                print("Shakestep: {}".format(i))
                # energy = sim_sys.analysis.energy()
                # energy_data[i] = energy["total"]
                # print(energy_data[i])
                if i % 100 == 0:

                    # write_dip_pos_file(sim_sys, dippos_file_name, rtype, i, rpids)
                    #s, fname, time, real_part_ids, fl, bonds=[]
                    wq.put({"func": write_vtk_file,"args": {"s": sim_sys, "fname": dippos_file_name,
                                   "time": i, "real_part_ids": sys_topology.rpids, "fl": ext_params.fl,
                                   "bonds": sys_topology.connected_reals}})
                    #write_vtk_file({"s": sim_sys, "fname": dippos_file_name,
                    #               "time": i, "real_part_ids": sys_topology.rpids, "fl": ext_params.fl,
                    #               "bonds": sys_topology.connected_reals})

            state = "p3m_setup"
        except KeyboardInterrupt:  # it is assumed that kill-signal from slurm is the SIGINT
            #with open(FINISH_FLAG_NAME, 'w') as flag_file:
            #    flag_file.write('{}'.format(i))
            wq.put({"func": raise_finish_flag, "args":{"filename": FINISH_FLAG_NAME,"data": '{}'.format(i)}})
            #raise_finish_flag({"filename": FINISH_FLAG_NAME,"data": '{}'.format(i)})
            steps = i
            checkpoint.save()
            print(
                "It is seems to me that it is high time to restart. State is {}"
                .format(state))

    if state == "p3m_setup":
        sim_sys.thermostat.set_langevin(kT=1, gamma=1.0, seed=ext_params.seed)

        sim_sys.periodicity = [1, 1, 1]
        p3m = magnetostatics.DipolarP3M(prefactor=1, mesh=32, accuracy=1E-3)
        sim_sys.actors.add(p3m)
        sim_sys.non_bonded_inter[rtype, rtype].lennard_jones.set_params(
            epsilon=_epsilon_attr, sigma=_sigma, cutoff=_r_cut_attr, shift=0)

        checkpoint.register("p3m")
        state = "production_loop"
        checkpoint.save()
    if state == "production_loop":
        sim_sys.periodicity = [1, 1, 1]
        sim_sys.time_step = 0.001
        try:
            for i in range(main_steps, main_run_steps):
                sim_sys.integrator.run(100)

                if i % 200 == 0:  # write somthing
                    print("Main step: {}".format(i))
                    # energy = sim_sys.analysis.energy()
                    # energy_data[i] = energy["total"]
                    # print(energy_data[i])

                    idx_file = equilibrium_steps + i
                    # write_dip_pos_file(sim_sys, dippos_file_name, rtype, idx_file, rpids)
                    #s, fname, time, real_part_ids, fl, bonds=[]
                    wq.put({"func": write_vtk_file,"args": {"s": sim_sys, "fname": dippos_file_name,"time": idx_file, "real_part_ids": sys_topology.rpids, "fl": ext_params.fl,
                                   "bonds": sys_topology.connected_reals}})
                    #write_vtk_file({"s": sim_sys, "fname": dippos_file_name,"time": idx_file, "real_part_ids": sys_topology.rpids, "fl": ext_params.fl,
                    #               "bonds": sys_topology.connected_reals})

        except KeyboardInterrupt:  # it is assumed that kill-signal from slurm is the SIGINT
            #with open(FINISH_FLAG_NAME, 'w') as flag_file:
            #    flag_file.write('{}'.format(i))
            wq.put({"func": raise_finish_flag, "args":{"filename": FINISH_FLAG_NAME,"data": '{}'.format(i)}})
            #raise_finish_flag({"filename": FINISH_FLAG_NAME,"data": '{}'.format(i)})
            main_steps = i
            print(sim_sys)
            checkpoint.save()
            print(("It is seems to me that",
                   "it is high time to restart. State is {}").format(state))
        else:
            wq.put({"func": raise_finish_flag, "args":{"filename": FINISH_FLAG_NAME,"data": "false"}})
            #raise_finish_flag({"filename": FINISH_FLAG_NAME,"data": "false"})
            checkpoint.save()
            print("Done")

    #end of prog
    print("main process end")

    #signalyze to writing process to end
    check.value=0
    p.join()

    print("sub process end")
