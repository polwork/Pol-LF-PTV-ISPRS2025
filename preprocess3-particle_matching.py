import numpy as np
from scipy.spatial import KDTree
from itertools import combinations, product
from pandas import read_csv

class match_blob_files:
    def __init__(self, blob_fnames, img_system, RIO, voxel_size, max_blob_dist,
                 max_err=1e9, reverse_eta_zeta=False, travered_voxel_rep=True):
        self.blobs = [np.array(read_csv(fn, sep='\t', header=None)) for fn in blob_fnames]
        self.imsys = img_system
        self.RIO = RIO
        self.voxel_size = voxel_size
        self.reverse_eta_zeta = reverse_eta_zeta
        self.max_blob_dist = max_blob_dist
        self.max_err = max_err
        self.time_lst = sorted(set(np.concatenate([bl[:, -1] for bl in self.blobs])))
        self.cam_names = [cam.name for cam in self.imsys.cameras]
        self.travered_voxel_rep = travered_voxel_rep

    def get_particles_dic(self, frame):
        pd = {}
        for i, cam_name in enumerate(self.cam_names):
            arr = self.blobs[i][self.blobs[i][:, -1] == frame]
            pd[cam_name] = arr[:, 1::-1].tolist() if self.reverse_eta_zeta else arr[:, :2].tolist()
        return pd

    def get_particles(self, frames=None):
        frames = self.time_lst if frames is None else frames
        self.particles = []
        previous_particles = []

        for tm in frames:
            pd = self.get_particles_dic(tm)
            nb = np.mean([len(pd[k]) for k in pd])

            if previous_particles:
                mut = matching_using_time(self.imsys, pd, previous_particles, self.max_err)
                mut.triangulate_candidates()
                self.particles.extend(p + [tm] for p in mut.matched_particles)
                pd = mut.return_updated_particle_dict()

            M = matching(self.imsys, pd, self.RIO, self.voxel_size, self.max_err)
            M.get_voxel_dictionary()
            M.list_candidates()
            M.get_particles()
            self.particles.extend(p + [tm] for p in M.matched_particles)

            if self.travered_voxel_rep:
                dv = self.voxel_size / 2.0
                new_ROI = [(self.RIO[i][0] + dv, self.RIO[i][1] - dv) for i in range(3)]
                new_pd = {k: [xy for xy in v if xy != [-1, -1]] for k, v in pd.items()}
                
                M2 = matching(self.imsys, new_pd, new_ROI, self.voxel_size, self.max_err)
                M2.get_voxel_dictionary()
                M2.list_candidates()
                M2.get_particles()
                self.particles.extend(p + [tm] for p in M2.matched_particles)

            previous_particles = [p for p in self.particles if p[-1] == tm]
        
        self.particles = [p for p in self.particles if p[4] < self.max_err]

    def save_results(self, fname):
        particles_to_save = []
        for p in self.particles:
            rd = dict(p[3])
            particle_data = p[:3] + [rd.get(i, [-1])[0] for i in range(len(self.imsys.cameras))] + [p[4], p[5]]
            particles_to_save.append(particle_data)
        np.savetxt(fname, particles_to_save, delimiter='\t')

class matching:
    def __init__(self, img_system, particles_dic, RIO, voxel_size, max_err=None):
        self.imsys = img_system
        self.RIO = RIO
        self.voxel_size = voxel_size
        self.max_err = max_err
        self.rays = []
        self.ray_camera_indexes = [0]
        
        for i, cam in enumerate(self.imsys.cameras):
            particles_i = particles_dic[cam.name]
            self.ray_camera_indexes.append(len(particles_i) + self.ray_camera_indexes[-1])
            for j, (x, y) in enumerate(particles_i):
                r_ij = cam.get_r(x, y)
                self.rays.append((x, y, (i, j), r_ij))
        
        self._init_voxel_grid()

    def _init_voxel_grid(self):
        self.voxels = []
        for axis in range(3):
            start, end = self.RIO[axis]
            length = end - start
            n_voxels = int(np.ceil(length / self.voxel_size))
            center = (start + end) / 2.0
            offset = (n_voxels // 2) * self.voxel_size if n_voxels % 2 else (n_voxels/2 - 0.5)*self.voxel_size
            self.voxels.append([i*self.voxel_size + center - offset for i in range(n_voxels)])

    def ray_traversed_voxels(self, ray):
        if ray[0] == -1 and ray[1] == -1: return []
        cam = self.imsys.cameras[ray[2][0]]
        O, r_ = cam.O, ray[3]
        a1, a2 = sorted([(self.RIO[2][0]-O[2])/r_[2], (self.RIO[2][1]-O[2])/r_[2]])
        ray_voxels = set()
        
        a = a1
        while a <= a2:
            pos = O + r_ * a
            if not all(self.RIO[i][0] <= pos[i] <= self.RIO[i][1] for i in range(2)): 
                a += self.voxel_size/4.0
                continue
            indices = tuple(int((pos[i]-self.voxels[i][0])//self.voxel_size) for i in range(3))
            ray_voxels.add((indices, ray[2]))
            a += self.voxel_size/4.0
        return ray_voxels

    def get_voxel_dictionary(self):
        self.voxel_dic = {}
        for ray in self.rays:
            for vxl in self.ray_traversed_voxels(ray):
                self.voxel_dic.setdefault(vxl[0], set()).add(vxl[1])

    def list_candidates(self):
        self.candidate_dic = {k: [] for k in range(2, len(self.imsys.cameras)+1)}
        for vxl, rays in self.voxel_dic.items():
            if len(rays) < 2: continue
            ray_by_cams = [[] for _ in self.imsys.cameras]
            for ray in rays: ray_by_cams[ray[0]].append(ray)
            
            for gs in self.candidate_dic:
                self.candidate_dic[gs].extend(product(*(rc for rc in ray_by_cams if rc)))

    def triangulate_rays(self, rays):
        camera_data = {}
        for ray in rays:
            cam_idx = ray[0]
            cam = self.imsys.cameras[cam_idx]
            camera_data[cam_idx] = (cam.O, self.rays[self.ray_camera_indexes[cam_idx]:self.ray_camera_indexes[cam_idx+1]][ray[1]][3])
        
        positions, distances = [], []
        for (i, (Oi, ri)), (j, (Oj, rj)) in combinations(camera_data.items(), 2):
            D, pos = line_dist(Oi, ri, Oj, rj)
            if self.max_err is None or D < 4*self.max_err:
                distances.append(D)
                positions.append(pos)
        return np.mean(positions, axis=0), list(camera_data.keys()), np.mean(distances)

    def get_particles(self):
        self.matched_particles = []
        used_rays = set()
        
        for gs in sorted(self.candidate_dic.keys(), reverse=True):
            candidates = [(cand, self.triangulate_rays(cand)) 
                          for cand in self.candidate_dic[gs] 
                          if not any(r in used_rays for r in cand)]
            candidates.sort(key=lambda x: x[1][2])
            
            for cand, (pos, _, err) in candidates:
                if err > self.max_err: continue
                blob_info = [(ri[0], (ri[1], self._get_eta_zeta(ri))) for ri in cand]
                self.matched_particles.append([*np.round(pos, 3), blob_info, np.round(err, 3)])
                used_rays.update(cand)

    def _get_eta_zeta(self, ray):
        cam_idx, blob_idx = ray
        ray_slice = self.rays[self.ray_camera_indexes[cam_idx]:self.ray_camera_indexes[cam_idx+1]]
        return ray_slice[blob_idx][0], ray_slice[blob_idx][1]

class matching_using_time:
    def __init__(self, img_system, particles_dic, previously_used_blobs, max_err=1e9):
        self.imsys = img_system
        self.pd = particles_dic
        self.prev_used_blobs = previously_used_blobs
        self.max_err = max_err
        self.trees = [KDTree(blobs) if blobs else None for blobs in self.pd.values()]

    def triangulate_candidates(self):
        triangulated = []
        for p in self.prev_used_blobs:
            candidate_blobs = {}
            for cam_idx, (_, (x, y)) in p[3]:
                if self.trees[cam_idx]:
                    candidate_blobs[cam_idx] = self.trees[cam_idx].data[self.trees[cam_idx].query((x, y))[1]]
            
            pos, err = self.imsys.stereo_match(candidate_blobs, self.max_err)[:2]
            if err < self.max_err:
                blob_info = [(ci, (i, tuple(blob))) for ci, blob in candidate_blobs.items()]
                triangulated.append([*np.round(pos, 3), blob_info, np.round(err, 3)])
        
        self.matched_particles = self._deduplicate_particles(triangulated)

    def _deduplicate_particles(self, particles):
        particles.sort(key=lambda x: x[-1])
        used_blobs, result = set(), []
        for p in particles:
            if all(blob not in used_blobs for _, blob in p[3]):
                result.append(p)
                used_blobs.update(blob for _, blob in p[3])
        return result

    def return_updated_particle_dict(self):
        new_pd = self.pd.copy()
        for p in self.matched_particles:
            for cam_idx, (_, (coords, _)) in p[3]:
                idx = np.where((new_pd[cam_idx] == coords).all(axis=1))[0]
                if idx.size: new_pd[cam_idx][idx[0]] = [-1, -1]
        return new_pd