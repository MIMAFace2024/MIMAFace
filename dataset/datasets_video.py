import os
import random
import numpy as np
import cv2
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
from threading import local
import struct
from io import BytesIO
from . import example_pb2
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

mean_face_lm5p_256 = np.array([
[(30.2946+8)*2+16, 51.6963*2],  # left eye pupil
[(65.5318+8)*2+16, 51.5014*2],  # right eye pupil
[(48.0252+8)*2+16, 71.7366*2],  # nose tip
[(33.5493+8)*2+16, 92.3655*2],  # left mouth corner
[(62.7299+8)*2+16, 92.2041*2],  # right mouth corner
], dtype=np.float32)


mean_box_lm4p_512 = np.array([
[80, 80], 
[80, 432], 
[432, 432], 
[432, 80],  
], dtype=np.float32)


def read_pts(pts_file):
    pts_num = 0
    with open(pts_file, 'r') as pts:
        line = pts.readlines()[1]
        pts_num = int(line[:-1].split(":")[1])
    with open(pts_file, 'r') as pts:
        lines = pts.readlines()[3:pts_num+3]
        pt = []
        for line in lines:
            pt.append(line.strip('\n').split(' '))
        pt = np.array(pt, dtype='float32')
    return pt


def get_lmp5(landmark_np):
    if landmark_np.shape[0] == 256 or landmark_np.shape[0] == 215 or landmark_np.shape[0] == 224 or landmark_np.shape[0] == 228:
        lefteye = (landmark_np[32:33, :] + landmark_np[44:45, :]) / 2
        righteye = (landmark_np[56:57, :] + landmark_np[68:69, :]) / 2
        nose = landmark_np[80:81]
        leftm = landmark_np[102:103]
        rightm = landmark_np[120:121]
    elif landmark_np.shape[0] == 94:
        lefteye = (landmark_np[16:17, :]+landmark_np[20:21, :])/2
        righteye = (landmark_np[24:25, :]+landmark_np[28:29, :])/2
        nose = landmark_np[32:33]
        leftm = landmark_np[45:46]
        rightm = landmark_np[51:52]
    elif landmark_np.shape[0] == 5:
        return landmark_np
    else:
        print('not supported!')
        return None

    return np.concatenate([lefteye, righteye, nose, leftm, rightm], axis=0)


def get_box_lm4p(pts):
    x1 = np.min(pts[:,0])
    x2 = np.max(pts[:,0])
    y1 = np.min(pts[:,1])
    y2 = np.max(pts[:,1])
    
    x_center = (x1+x2)*0.5
    y_center = (y1+y2)*0.5
    box_size = max(x2-x1, y2-y1)
    
    x1 = x_center-0.5*box_size
    x2 = x_center+0.5*box_size
    y1 = y_center-0.5*box_size
    y2 = y_center+0.5*box_size

    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)


def get_affine_transform(target_face_lm5p, mean_lm5p):
    mat_warp = np.zeros((2,3))
    A = np.zeros((4,4))
    B = np.zeros((4))
    for i in range(5):
        #sa[0][0] += a[i].x*a[i].x + a[i].y*a[i].y;
        A[0][0] += target_face_lm5p[i][0] * target_face_lm5p[i][0] + target_face_lm5p[i][1] * target_face_lm5p[i][1]
        #sa[0][2] += a[i].x;
        A[0][2] += target_face_lm5p[i][0]
        #sa[0][3] += a[i].y;
        A[0][3] += target_face_lm5p[i][1]

        #sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
        B[0] += target_face_lm5p[i][0] * mean_lm5p[i][0] + target_face_lm5p[i][1] * mean_lm5p[i][1]
        #sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
        B[1] += target_face_lm5p[i][0] * mean_lm5p[i][1] - target_face_lm5p[i][1] * mean_lm5p[i][0]
        #sb[2] += b[i].x;
        B[2] += mean_lm5p[i][0]
        #sb[3] += b[i].y;
        B[3] += mean_lm5p[i][1]

    #sa[1][1] = sa[0][0];
    A[1][1] = A[0][0]
    #sa[2][1] = sa[1][2] = -sa[0][3];
    A[2][1] = A[1][2] = -A[0][3]
    #sa[3][1] = sa[1][3] = sa[2][0] = sa[0][2];
    A[3][1] = A[1][3] = A[2][0] = A[0][2]
    #sa[2][2] = sa[3][3] = count;
    A[2][2] = A[3][3] = 5
    #sa[3][0] = sa[0][3];
    A[3][0] = A[0][3]

    _, mat23 = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    mat_warp[0][0] = mat23[0]
    mat_warp[1][1] = mat23[0]
    mat_warp[0][1] = -mat23[1]
    mat_warp[1][0] = mat23[1]
    mat_warp[0][2] = mat23[2]
    mat_warp[1][2] = mat23[3]

    return mat_warp



def hflip256(pts256, w):
    tmp_pts = pts256.copy()
    tmp_pts[:,0] = w - 1 - tmp_pts[:,0] 
    flip_pts = np.concatenate((tmp_pts[16:32,:], tmp_pts[0:16,:], tmp_pts[56:80,:], tmp_pts[32:56,:], 
    tmp_pts[80:85,:], np.flip(tmp_pts[91:98, :],0), np.flip(tmp_pts[85:91, :],0), tmp_pts[100:102,:], tmp_pts[98:100,:], 
    np.flip(tmp_pts[102:121],0), np.flip(tmp_pts[121:138],0),np.flip(tmp_pts[138:157],0),np.flip(tmp_pts[157:174],0), 
    np.flip(tmp_pts[174:215],0), np.flip(tmp_pts[215:222],0), np.flip(tmp_pts[222:224],0), tmp_pts[240:256,:], tmp_pts[224:240,:]), axis=0)
    
    return flip_pts


def transformation_from_points(points1, points2):
    points1 = np.float64(np.matrix([[point[0], point[1]] for point in points1]))
    points2 = np.float64(np.matrix([[point[0], point[1]] for point in points2]))

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    #points2 = np.array(points2)
    #write_pts('pt2.txt', points2)
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.array(np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])[:2])


def draw_pts(img, pts, mode="pts", shift=4, color=(255,255, 0), radius=1, thickness=1, save_path=None, dif=0):
    for cnt,p in enumerate(pts):
        if mode == "index":
            cv2.circle(img, (int(p[0] * (1 << shift)), int(p[1] * (1 << shift))), radius << shift, color, -1, cv2.LINE_AA, shift=shift)
            cv2.putText(img, str(cnt), (int(float(p[0] + dif)), int(float(p[1] + dif))),cv2.FONT_HERSHEY_SIMPLEX, radius/3, (0,0,255),thickness)
        elif mode == 'pts':
            cv2.circle(img, (int(p[0] * (1<< shift)),int(p[1] * (1<< shift))) ,radius<<shift, color, -1, cv2.LINE_AA, shift=shift)
            #cv2.circle(img, (int(p[0]),int(p[1])) ,1, color,1)
        else:
            print ('not support mode!')
            return
    if(save_path!=None):
        cv2.imwrite(save_path,img)
    return img


def vis_parsing_maps(im_cv2, parsing_anno_cv2, num_class=2):
    # Colors for all 20 parts
    part_colors = np.array([[255, 0, 0], [255, 255, 255], [255, 170, 0],
                            [255, 0, 85], [255, 0, 170],
                            [0, 255, 0], [85, 255, 0], [170, 255, 0],
                            [0, 255, 85], [0, 255, 170],
                            [0, 0, 255], [85, 0, 255], [170, 0, 255],
                            [0, 85, 255], [0, 170, 255],
                            [255, 255, 0], [255, 255, 85], [255, 255, 170],
                            [255, 0, 255], [255, 85, 255], [255, 170, 255],
                            [0, 255, 255], [85, 255, 255], [170, 255, 255]]).reshape((-1, 3)).astype(np.uint8)

    h, w, c = im_cv2.shape
    vis_parsing_anno = cv2.resize(parsing_anno_cv2, (w, h), interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros_like(im_cv2)

    num_of_class = num_class

    # print(im.shape)
    # print(vis_parsing_anno.shape)
    if num_of_class == 2:
        vis_parsing_anno_color = 255 * vis_parsing_anno[:, :, np.newaxis].repeat(3, axis=2)
    else:
        for pi in range(1, num_of_class):
            index = (vis_parsing_anno == pi)  # [:, :, np.newaxis].repeat(3, axis=2)
            vis_parsing_anno_color[index, :] = part_colors[pi]

    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(im_cv2, 0.4, vis_parsing_anno_color, 0.6, 0)
    # Save result or not
    return vis_im
 


class DatasetFaceDrive(Dataset):
    def __init__(self, root_dir, list_path, image_size, clip_processor, train=False):
        super(DatasetFaceAdapterDrive, self).__init__()

        self.root_dir = root_dir
        self.train = train
        self.clip_processor = clip_processor
        
        self.transform_aug = transforms.Compose([ 
            transforms.ColorJitter(0.3,0.3,0.3), 
            transforms.RandomGrayscale(p=0.01)])

        self.transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize(mean=0.5, std=0.5)])
        
        self.image_size = image_size

        self.thread_local = local()
        # every thread keep an instance
        self.thread_local.cache = {}

        temp_samples = []
   
        id_list_dict = {}
        id_index = 0

        sample_idx = 0
        with open(list_path, 'r') as f:
            for line in f.readlines():
                line_info = line.strip().split(' ')
                tffile, offset, im_id = line_info[0:3]
                if im_id=='ffhq_no_hair':
                    continue
                # im_id = im_id.replace('_no_hair', '_hair')
                temp_samples.append([tffile, offset, im_id])

                if im_id not in id_list_dict.keys():
                    id_list_dict[im_id] = [sample_idx]
                    id_index += 1
                else:
                    id_list_dict[im_id].append(sample_idx)
                    
                sample_idx += 1
                # if sample_idx>1000:
                #     break
                
                # if tffile.find('arkit_exp_bernice_wallace')>0:
                #     for _ in range(10):
                #         temp_samples.append([tffile, offset, im_id])
                #         sample_idx += 1

        self.samples = temp_samples
        self.samples_num = sample_idx
        print('samples : {}'.format(sample_idx))

        self.id_list_dict = id_list_dict
        self.label_num = id_index
        print('id label_num', id_index)

    def get_record(self, f, offset):
        f.seek(offset)

        # length,crc
        byte_len_crc = f.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        # proto,crc
        pb_data = f.read(proto_len)
        if len(pb_data) != proto_len:
            print("read pb_data err,proto_len:%s pb_data len:%s" %
                  (proto_len, len(pb_data)))
            return None

        example = example_pb2.Example()
        example.ParseFromString(pb_data)

        return example.features.feature
    
    def get_im_mask_pts(self, tffile, offset):
        tfrecord_file_path = os.path.join(self.root_dir, tffile)
        offset = int(offset)
        # every thread keep a f instace
        f = self.thread_local.cache.get(tfrecord_file_path, None)
        if f is None:
            f = open(tfrecord_file_path, 'rb')
            self.thread_local.cache[tfrecord_file_path] = f
        feature = self.get_record(f, offset)


        image_raw = feature['image'].bytes_list.value[0]
        image = Image.open(BytesIO(image_raw)).convert('RGB')
        w,h = image.size
 
        image_raw = feature['mask'].bytes_list.value[0]
        mask = Image.open(BytesIO(image_raw)).convert('L').resize((w, h), Image.NEAREST)

        if 'pts256' in feature.keys():
            landmarks_data = feature['pts256'].bytes_list.value[0]
            lm_num=256
            lm_list = landmarks_data.decode().split('\n')[3:3+lm_num]
        else:
            print('error, no pts found!')

        pts = np.asarray([[line.split()[0], line.split()[1]] for line in lm_list], dtype=np.float32)
        
        return image, mask, pts

    def __getitem__(self, index):

        tffile, offset, im_id = self.samples[index]
        image_src, mask_src, pts_src = self.get_im_mask_pts(tffile, offset)
        
        same_id_list = self.id_list_dict[im_id]
        rand_idx = random.sample(same_id_list, 1)[0]
        tffile, offset, im_id = self.samples[rand_idx]
        image_tar, mask_tar, pts_tar = self.get_im_mask_pts(tffile, offset)
        
        box_pts4 = get_box_lm4p(np.concatenate([pts_src, pts_tar], axis=0))
        if self.train:
            target_face_size = random.randint(self.image_size//2+32, self.image_size-80)
            x1 = random.randint(32, self.image_size-32-target_face_size)
            y1 = random.randint(32, self.image_size-32-target_face_size)
            x2 = x1+target_face_size
            y2 = y1+target_face_size
            x1 = x1+random.randint(-32,32)
            x2 = x2+random.randint(-32,32)
            y1 = y1+random.randint(-32,32)
            y2 = y2+random.randint(-32,32)
            random_mean_box_lm4p_512 = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)
        else:
            random_mean_box_lm4p_512=mean_box_lm4p_512
        warp_mat = transformation_from_points(box_pts4, random_mean_box_lm4p_512 * self.image_size/512)

        image_crop_src = cv2.warpAffine(np.array(image_src), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
        mask_crop_src = cv2.warpAffine(np.array(mask_src), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
        pts_crop_src = np.concatenate((pts_src, np.ones((pts_src.shape[0],1), dtype=np.float32)), axis=1).dot(warp_mat.T)
        # image_crop = self.transform_aug(image_crop)
        # if random.randint(0,1)==1:
        #     image_crop = image_crop.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask_crop = cv2.flip(mask_crop, 1)
        #     pts_crop = hflip256(pts_crop, self.image_size)
        pts5 = get_lmp5(pts_crop_src)
        warp_mat_256_src = get_affine_transform(pts5, mean_face_lm5p_256)
        image_src_crop256 = cv2.warpAffine(image_crop_src, warp_mat_256_src, (256, 256), flags=cv2.INTER_LINEAR)
        image_src_crop256 = Image.fromarray(image_src_crop256)
        if self.train:
            image_src_crop256 = self.transform_aug(image_src_crop256)
        image_crop_src = Image.fromarray(image_crop_src)
        image_src_for_clip = self.clip_processor(images=image_crop_src, return_tensors="pt").pixel_values.view(3, self.clip_processor.crop_size["height"], self.clip_processor.crop_size["width"])
        
        if self.train:
            target_face_size = random.randint(self.image_size//2+32, self.image_size-80)
            x1 = random.randint(32, self.image_size-32-target_face_size)
            y1 = random.randint(32, self.image_size-32-target_face_size)
            x2 = x1+target_face_size
            y2 = y1+target_face_size
            x1 = x1+random.randint(-32,32)
            x2 = x2+random.randint(-32,32)
            y1 = y1+random.randint(-32,32)
            y2 = y2+random.randint(-32,32)
            random_mean_box_lm4p_512 = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)

            warp_mat = transformation_from_points(box_pts4, random_mean_box_lm4p_512 * self.image_size/512)
            
        image_crop_tar = cv2.warpAffine(np.array(image_tar), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
        mask_crop_tar = cv2.warpAffine(np.array(mask_tar), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
        pts_crop_tar = np.concatenate((pts_tar, np.ones((pts_tar.shape[0],1), dtype=np.float32)), axis=1).dot(warp_mat.T)
        pts5_tar = get_lmp5(pts_crop_tar)
        warp_mat_256_tar = get_affine_transform(pts5_tar, mean_face_lm5p_256)
        image_tar_crop256 = cv2.warpAffine(image_crop_tar, warp_mat_256_tar, (256, 256), flags=cv2.INTER_LINEAR)
        image_tar_crop256 = Image.fromarray(image_tar_crop256)
        image_crop_tar = Image.fromarray(image_crop_tar)
        image_tar_for_clip = self.clip_processor(images=image_crop_tar, return_tensors="pt").pixel_values.view(3, self.clip_processor.crop_size["height"], self.clip_processor.crop_size["width"])
        # if random.randint(0,1)==1:
        #     image_src_crop256 = image_src_crop256.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask_src_crop256 = cv2.flip(mask_src_crop256, 1) 
        
        sample = {
                  'image_src': self.transform(image_crop_src),
                  'mask_src': torch.from_numpy(mask_crop_src),
                  'image_src_clip': image_src_for_clip,
                  'image_src_crop256': self.transform(image_src_crop256),
                  'image_tar': self.transform(image_crop_tar), 
                  'mask_tar': torch.from_numpy(mask_crop_tar),
                  'image_tar_clip': image_tar_for_clip,
                  'image_tar_crop256': self.transform(image_tar_crop256), 
                  'image_tar_warpmat256': torch.from_numpy(warp_mat_256_tar),
                  'image_src_warpmat256': torch.from_numpy(warp_mat_256_src),
                  'image_tar_inverse_warpmat256': torch.from_numpy(cv2.invertAffineTransform(warp_mat_256_tar)),
                  'image_src_inverse_warpmat256': torch.from_numpy(cv2.invertAffineTransform(warp_mat_256_src))
                  }
        return sample

    def __len__(self):
        return len(self.samples)
        

class DatasetDriveVideo(Dataset):
    def __init__(self, root_dir, list_path, image_size, clip_processor, train=True, video_length=8):
        super(DatasetDriveVideo, self).__init__()

        self.root_dir = root_dir
        self.train = train
        self.clip_processor = clip_processor
        self.video_length = video_length
        
        self.transform_aug = transforms.Compose([ 
            transforms.ColorJitter(0.3,0.3,0.3), 
            transforms.RandomGrayscale(p=0.01)])

        self.transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize(mean=0.5, std=0.5)])
        
        self.image_size = image_size

        self.thread_local = local()
        # every thread keep an instance
        self.thread_local.cache = {}

        temp_samples = []
        temp_sample_idxs = []
        total_samples = []
   
        id_list_dict = {}
        id_index = 0

        sample_idx = 0
        current_id = None
        with open(list_path, 'r') as f:
            for line in f.readlines():
                line_info = line.strip().split(' ')
                tffile, offset, im_id = line_info[0:3]
                
                if im_id != current_id:
                    if len(temp_sample_idxs)>video_length:
                        total_samples += temp_samples
                        id_list_dict[current_id] = temp_sample_idxs
                        id_index += 1 
                    
                    current_id = im_id
                    sample_idx = len(total_samples)
                    temp_samples = [[tffile, offset, im_id]]
                    temp_sample_idxs = [sample_idx]
                    sample_idx += 1
                    
                else:
                    temp_samples.append([tffile, offset, im_id])
                    temp_sample_idxs.append(sample_idx)
                    sample_idx += 1


        self.samples = total_samples
        self.samples_num = sample_idx
        print('samples : {}'.format(sample_idx))

        self.id_list_dict = id_list_dict
        self.label_num = id_index
        print('id label_num', id_index)

    def get_record(self, f, offset):
        f.seek(offset)

        # length,crc
        byte_len_crc = f.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        # proto,crc
        pb_data = f.read(proto_len)
        if len(pb_data) != proto_len:
            print("read pb_data err,proto_len:%s pb_data len:%s" %
                  (proto_len, len(pb_data)))
            return None

        example = example_pb2.Example()
        example.ParseFromString(pb_data)

        return example.features.feature
    
    def get_im_mask_pts(self, tffile, offset):
        tfrecord_file_path = os.path.join(self.root_dir, tffile)
        offset = int(offset)
        # every thread keep a f instace
        f = self.thread_local.cache.get(tfrecord_file_path, None)
        if f is None:
            f = open(tfrecord_file_path, 'rb')
            self.thread_local.cache[tfrecord_file_path] = f
        feature = self.get_record(f, offset)


        image_raw = feature['image'].bytes_list.value[0]
        image = Image.open(BytesIO(image_raw)).convert('RGB')
        w,h = image.size
 
        image_raw = feature['mask'].bytes_list.value[0]
        mask = Image.open(BytesIO(image_raw)).convert('L').resize((w, h), Image.NEAREST)

        if 'pts256' in feature.keys():
            landmarks_data = feature['pts256'].bytes_list.value[0]
            lm_num=256
            lm_list = landmarks_data.decode().split('\n')[3:3+lm_num]
        else:
            print('error, no pts found!')

        pts = np.asarray([[line.split()[0], line.split()[1]] for line in lm_list], dtype=np.float32)
        
        return image, mask, pts

    def __getitem__(self, index):

        tffile, offset, im_id = self.samples[index]
        image_src, mask_src, pts_src = self.get_im_mask_pts(tffile, offset)
        
        same_id_list = self.id_list_dict[im_id]
        cur_video_length = len(same_id_list)
        start_idx = random.randint(0, cur_video_length-self.video_length)
        video_clip_idxs = same_id_list[start_idx:start_idx+self.video_length]
        
        
        tffile, offset, im_id = self.samples[video_clip_idxs[0]]
        image_tar, mask_tar, pts_tar = self.get_im_mask_pts(tffile, offset)
        
        box_pts4 = get_box_lm4p(np.concatenate([pts_src, pts_tar], axis=0))
        if self.train:
            target_face_size = random.randint(self.image_size//2, self.image_size-80)
            x1 = random.randint(32, self.image_size-32-target_face_size)
            y1 = random.randint(32, self.image_size-32-target_face_size)
            x2 = x1+target_face_size
            y2 = y1+target_face_size
            x1 = x1+random.randint(-16,16)
            x2 = x2+random.randint(-16,16)
            y1 = y1+random.randint(-16,16)
            y2 = y2+random.randint(-16,16)
            random_mean_box_lm4p_512 = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)
        else:
            random_mean_box_lm4p_512=mean_box_lm4p_512
        warp_mat_share = transformation_from_points(box_pts4, random_mean_box_lm4p_512 * self.image_size/512)
        
        image_crop_tar = cv2.warpAffine(np.array(image_tar), warp_mat_share, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
        # mask_crop_tar = cv2.warpAffine(np.array(mask_tar), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
        pts_crop_tar = np.concatenate((pts_tar, np.ones((pts_tar.shape[0],1), dtype=np.float32)), axis=1).dot(warp_mat_share.T)
        pts5_tar = get_lmp5(pts_crop_tar)
        warp_mat_256_tar = get_affine_transform(pts5_tar, mean_face_lm5p_256)
        image_tar_crop256 = cv2.warpAffine(image_crop_tar, warp_mat_256_tar, (256, 256), flags=cv2.INTER_LINEAR)
        image_tar_crop256 = Image.fromarray(image_tar_crop256)
        image_crop_tar = Image.fromarray(image_crop_tar)
        # image_tar_for_clip = self.clip_processor(images=image_crop_tar, return_tensors="pt").pixel_values.view(3, 224, 224)
        
        warp_mat_256_tar_list = [torch.from_numpy(warp_mat_256_tar).view(1,2,3)]
        images_tar_crop256_list = [self.transform(image_tar_crop256).view(1,3,256,256)]
        images_tar_list = [self.transform(image_crop_tar).view(1,3,self.image_size, self.image_size)]
        inverse_warp_mat_256_tar_list = [cv2.invertAffineTransform(warp_mat_256_tar)]
        for frame_idx in video_clip_idxs[1:]:
            tffile, offset, im_id = self.samples[frame_idx]
            image_tar, mask_tar, pts_tar = self.get_im_mask_pts(tffile, offset)
            image_crop_tar = cv2.warpAffine(np.array(image_tar), warp_mat_share, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
            # mask_crop_tar = cv2.warpAffine(np.array(mask_tar), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
            pts_crop_tar = np.concatenate((pts_tar, np.ones((pts_tar.shape[0],1), dtype=np.float32)), axis=1).dot(warp_mat_share.T)
            pts5_tar = get_lmp5(pts_crop_tar)
            warp_mat_256_tar = get_affine_transform(pts5_tar, mean_face_lm5p_256)
            image_tar_crop256 = cv2.warpAffine(image_crop_tar, warp_mat_256_tar, (256, 256), flags=cv2.INTER_LINEAR)
            image_tar_crop256 = Image.fromarray(image_tar_crop256)
            image_crop_tar = Image.fromarray(image_crop_tar)
            
            warp_mat_256_tar_list.append(torch.from_numpy(warp_mat_256_tar).view(1,2,3))
            inverse_warp_mat_256_tar_list.append(cv2.invertAffineTransform(warp_mat_256_tar))
            images_tar_crop256_list.append(self.transform(image_tar_crop256).view(1,3,256,256))
            images_tar_list.append(self.transform(image_crop_tar).view(1,3,self.image_size, self.image_size))
        
        
        images_tar = torch.cat(images_tar_list, dim=0)
        images_tar_crop256 = torch.cat(images_tar_crop256_list, dim=0)
        warp_mat_256_tar = torch.cat(warp_mat_256_tar_list, dim=0)
        inverse_warp_mat_256_tar = np.stack(inverse_warp_mat_256_tar_list, axis=0)

        image_crop_src = cv2.warpAffine(np.array(image_src), warp_mat_share, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
        # mask_crop_src = cv2.warpAffine(np.array(mask_src), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
        pts_crop_src = np.concatenate((pts_src, np.ones((pts_src.shape[0],1), dtype=np.float32)), axis=1).dot(warp_mat_share.T)

        pts5 = get_lmp5(pts_crop_src)
        warp_mat_256_src = get_affine_transform(pts5, mean_face_lm5p_256)
        image_src_crop256 = cv2.warpAffine(image_crop_src, warp_mat_256_src, (256, 256), flags=cv2.INTER_LINEAR)
        image_src_crop256 = Image.fromarray(image_src_crop256)
        if self.train:
            image_src_crop256 = self.transform_aug(image_src_crop256)
        image_crop_src = Image.fromarray(image_crop_src)
        image_src_for_clip = self.clip_processor(images=image_crop_src, return_tensors="pt").pixel_values.view(3, 224, 224)
        
        # use same rect as ref 
        # if self.train:
        #     random_mean_box_lm4p_512=random_mean_box_lm4p_512+(np.random.rand(*random_mean_box_lm4p_512.shape)*2-1)*np.array([[16, 16]])
        #     warp_mat = transformation_from_points(box_pts4, random_mean_box_lm4p_512 * self.image_size/512)
        # image_crop_tar = cv2.warpAffine(np.array(image_tar), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
        # mask_crop_tar = cv2.warpAffine(np.array(mask_tar), warp_mat, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
        # pts_crop_tar = np.concatenate((pts_tar, np.ones((pts_tar.shape[0],1), dtype=np.float32)), axis=1).dot(warp_mat.T)
        # pts5_tar = get_lmp5(pts_crop_tar)
        # warp_mat_256_tar = get_affine_transform(pts5_tar, mean_face_lm5p_256)
        # image_tar_crop256 = cv2.warpAffine(image_crop_tar, warp_mat_256_tar, (256, 256), flags=cv2.INTER_LINEAR)
        # image_tar_crop256 = Image.fromarray(image_tar_crop256)
        # image_crop_tar = Image.fromarray(image_crop_tar)
        # image_tar_for_clip = self.clip_processor(images=image_crop_tar, return_tensors="pt").pixel_values.view(3, 224, 224)
        # if random.randint(0,1)==1:
        #     image_src_crop256 = image_src_crop256.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask_src_crop256 = cv2.flip(mask_src_crop256, 1) 
        
        sample = {
                  'image_src': self.transform(image_crop_src),
                #   'mask_src': torch.from_numpy(mask_crop_src),
                  'image_src_clip': image_src_for_clip,
                  'image_src_crop256': self.transform(image_src_crop256),
                  'image_src_warpmat256': warp_mat_256_src,
                  'image_src_inverse_warpmat256': cv2.invertAffineTransform(warp_mat_256_src),
                  'image_tar': images_tar, 
                #   'mask_tar': torch.from_numpy(mask_crop_tar),
                #   'image_tar_clip': image_tar_for_clip,
                  'image_tar_crop256': images_tar_crop256, 
                  'image_tar_warpmat256': warp_mat_256_tar,
                  'image_tar_inverse_warpmat256': inverse_warp_mat_256_tar,
                  }
        return sample

    def __len__(self):
        return len(self.samples)
        
