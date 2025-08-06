import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.ndimage.measurements import label, find_objects
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import cv2

class ParticleSegmentation:
    def __init__(self, image, sigma=None, threshold=10, mask=1.0,
                 median=None, local_filter=None, particle_size=3,
                 min_xsize=None, max_xsize=None,
                 min_ysize=None, max_ysize=None,
                 min_mass=None, max_mass=None,
                 method='labeling'):
        self.im = image
        self.p_size = particle_size
        self.sigma = sigma
        self.median = median
        self.th = threshold
        self.mask = mask
        self.bbox_limits = (min_xsize, max_xsize, min_ysize, max_ysize)
        self.mass_limits = (min_mass, max_mass)
        self.loc_filter = local_filter
        self.method = method

    def local_filter(self, image):
        w = self.loc_filter
        window = np.ones((w, w)) / w**2
        local_mean = np.convolve2d(image, window, mode='same')
        new_im = image - local_mean
        new_im[new_im < 0] = 0
        return new_im.astype('uint8')

    def process_image(self):
        processed = self.im.copy()
        
        if self.sigma is not None:
            processed = gaussian_filter(processed, self.sigma)
        if self.median is not None:
            processed = median_filter(processed, size=self.median)
        if self.loc_filter is not None:
            processed = self.local_filter(processed)
        
        self.processed_im = processed * self.mask

    def get_binary_image(self):
        self.process_image()
        
        if self.method == 'labeling':
            return (self.processed_im > self.th) * self.mask
        elif self.method == 'dilation':
            from scipy.ndimage import grey_dilation
            dilated = grey_dilation(self.processed_im, size=self.p_size, mode='constant')
            return (self.processed_im == dilated) * (self.processed_im > self.th)

    def characterize_blob(self, coord, size=None):
        size = size or self.p_size
        r = size // 2
        y, x = coord
        
        # Calculate slices for blob neighborhood
        y_slice = slice(max(0, y - r), min(self.im.shape[0], y + r + 1))
        x_slice = slice(max(0, x - r), min(self.im.shape[1], x + r + 1))
        
        # Extract blob region
        region = self.processed_im[y_slice, x_slice]
        mass = region.sum()
        
        # Create coordinate grids
        yy, xx = np.mgrid[y_slice, x_slice]
        
        # Calculate center of mass
        cy = (yy * region).sum() / mass
        cx = (xx * region).sum() / mass
        
        # Calculate bounding box
        above_th = region > self.th
        if above_th.any():
            y_min, y_max = yy[above_th].min(), yy[above_th].max()
            x_min, x_max = xx[above_th].min(), xx[above_th].max()
            bbox = [y_max - y_min + 1, x_max - x_min + 1]
        else:
            bbox = [0, 0]
        
        return (cy, cx), bbox, mass

    def blob_labeling(self, image):
        labeled, num_features = label(image)
        locations = find_objects(labeled)
        self.labeled = labeled
        return locations

    def get_blobs(self):
        bin_im = self.get_binary_image()
        blobs = []
        
        if self.method == 'dilation':
            y_coords, x_coords = np.where(bin_im > 0)
            
            for y, x in zip(y_coords, x_coords):
                center = (y, x)
                for _ in range(3):  # Refine position through iterations
                    center, bbox, mass = self.characterize_blob(center)
                    if np.linalg.norm(np.array(center) - np.array((y, x))) < 1.0:
                        break
                
                blobs.append([center, bbox, mass])
            
            # Remove duplicates
            if blobs:
                points = [b[0] for b in blobs]
                tree = KDTree(points)
                duplicates = tree.query_pairs(self.p_size / 2)
                remove_indices = set()
                
                for i, j in duplicates:
                    if blobs[i][2] < blobs[j][2]:
                        remove_indices.add(j)
                    else:
                        remove_indices.add(i)
                
                blobs = [b for i, b in enumerate(blobs) if i not in remove_indices]
        
        elif self.method == 'labeling':
            blob_locations = self.blob_labeling(bin_im)
            
            for loc in blob_locations:
                region = self.labeled[loc]
                mask = (region > 0).astype(float)
                region_im = self.processed_im[loc] * mask
                mass = region_im.sum()
                
                # Create coordinate grids
                yy, xx = np.mgrid[loc[0], loc[1]]
                
                # Calculate center of mass
                cy = (yy * region_im).sum() / mass
                cx = (xx * region_im).sum() / mass
                center = (round(cy, 2), round(cx, 2))
                
                blobs.append([center, list(mask.shape), mass])
        
        self.blobs = blobs

    def apply_blobs_size_filter(self):
        min_x, max_x, min_y, max_y = self.bbox_limits
        min_m, max_m = self.mass_limits
        
        filtered = []
        for center, bbox, mass in self.blobs:
            size_y, size_x = bbox
            
            if min_x is not None and size_x < min_x:
                continue
            if max_x is not None and size_x > max_x:
                continue
            if min_y is not None and size_y < min_y:
                continue
            if max_y is not None and size_y > max_y:
                continue
            if min_m is not None and mass < min_m:
                continue
            if max_m is not None and mass > max_m:
                continue
            
            filtered.append([center, bbox, mass])
        
        self.blobs = filtered

    def save_results(self, fname):
        data = []
        for (y, x), (size_y, size_x), mass in self.blobs:
            data.append([y, x, size_y, size_x, mass, 0])
        
        np.savetxt(fname, data, fmt=['%.2f', '%.2f', '%d', '%d', '%.2f', '%d'], delimiter='\t')


class LoopSegmentation:
    def __init__(self, dir_name, extension='.tif', image_start=0, N_img=None,
                 sigma=1.0, threshold=10, mask=1.0, local_filter=15, median=None,
                 particle_size=3, min_xsize=None, max_xsize=None, min_ysize=None,
                 max_ysize=None, min_mass=None, max_mass=None, method='labeling'):
        self.dir_name = dir_name
        self.extension = extension
        self.image_start = image_start
        self.N_img = N_img
        self.seg_params = {
            'sigma': sigma,
            'threshold': threshold,
            'mask': mask,
            'median': median,
            'local_filter': local_filter,
            'particle_size': particle_size,
            'min_xsize': min_xsize,
            'max_xsize': max_xsize,
            'min_ysize': min_ysize,
            'max_ysize': max_ysize,
            'min_mass': min_mass,
            'max_mass': max_mass,
            'method': method
        }
        self.blobs = []

    def process_folder(self):
        import os
        from skimage.io import imread
        
        files = sorted([f for f in os.listdir(self.dir_name) if f.endswith(self.extension)])
        files = files[self.image_start:]
        
        if self.N_img is not None:
            files = files[:self.N_img]
        
        for frame, file in enumerate(files, self.image_start):
            img_path = os.path.join(self.dir_name, file)
            img = imread(img_path, as_gray=True)
            
            seg = ParticleSegmentation(img, **self.seg_params)
            seg.get_blobs()
            seg.apply_blobs_size_filter()
            
            for (y, x), (size_y, size_x), mass in seg.blobs:
                self.blobs.append([y, x, size_y, size_x, mass, frame])
            
            # Visualization (optional)
            vis_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for (y, x), (size_y, size_x), _ in seg.blobs:
                pt1 = (int(x - size_x/2), int(y - size_y/2))
                pt2 = (int(x + size_x/2), int(y + size_y/2))
                cv2.rectangle(vis_img, pt1, pt2, (0, 0, 255), 2)
            
            cv2.imwrite(f"output_frame_{frame}.png", vis_img)
        
        return self.blobs

    def save_results(self, fname):
        np.savetxt(fname, self.blobs, 
                   fmt=['%.2f', '%.2f', '%d', '%d', '%.2f', '%d'], 
                   delimiter='\t')